import os
from typing import TYPE_CHECKING, Union, cast, Optional, Tuple, List
import sys
import numpy as np
import open3d as o3d
import json
import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# 设置路径
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
submodules_dir = os.path.join(current_script_dir, '../habitat-lab-old/habitat-lab')
sys.path.append(submodules_dir)

# 静默日志
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# 设置CPU渲染环境变量
# os.environ["EGL_DEVICE_ID"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# # 不设置MAGNUM_DEVICE为cpu，让它自动选择合适的渲染后端
# # os.environ["MAGNUM_DEVICE"] = "cpu"
# os.environ["HABITAT_SIM_HEADLESS"] = "1"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class ShortestPathFollowerAgent(Agent):
    """最短路径跟随智能体"""
    
    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


class NavigationAPI:
    """Habitat导航API接口类
    
    提供基于点云和语义分割结果的导航功能，生成导航视频。
    """
    
    def __init__(self, 
                 yaml_path: str = "/root/sjm/habitat-lab-old/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_scannet.yaml",
                 scene_path: Optional[str] = None,
                 start_position: Optional[List[float]] = None,
                 goal_radius: float = 0.2,
                 fps: int = 6,
                 video_quality: int = 9):
        """
        初始化导航API
        
        Args:
            yaml_path: Habitat配置文件路径
            scene_path: 场景文件路径 (.glb文件)
            start_position: 起始位置坐标 [x, y, z]，如果为None则根据场景边界自动计算
            goal_radius: 目标半径
            fps: 视频帧率
            video_quality: 视频质量 (1-10)
        """
        self.yaml_path = yaml_path
        self.scene_path = scene_path
        self.start_position = start_position  # 不再设置默认值，将在运行时计算
        self.goal_radius = goal_radius
        self.fps = fps
        self.video_quality = video_quality
        
    def _load_point_cloud(self, ply_path: str) -> np.ndarray:
        """加载并预处理点云数据"""
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # 坐标系平移：将点云坐标系从以0为中心转换为从0开始
        x_offset = -points[:, 0].min()
        y_offset = -points[:, 1].min()
        z_offset = 0
        
        points[:, 0] += x_offset
        points[:, 1] += y_offset
        points[:, 2] += z_offset
        
        return points
    
    def _load_segmentation_mask(self, jsonl_path: str) -> np.ndarray:
        """加载语义分割掩码"""
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                return np.array(data["pred_mask"])
        raise ValueError("无法从JSONL文件中读取分割掩码")
    
    def _transform_coordinates(self, target_center: np.ndarray) -> np.ndarray:
        """坐标系转换：点云坐标系 -> Habitat坐标系"""
        # 点云: [x, y, z] -> Habitat: [x, z, -y]
        transformed = target_center.copy()
        transformed[1], transformed[2] = transformed[2], -transformed[1]
        return transformed
    
    def _create_habitat_config(self, scene_path: Optional[str] = None) -> habitat.config:
        """创建Habitat配置"""
        config = habitat.get_config(config_path=self.yaml_path)
        
        with habitat.config.read_write(config):
            # 如果指定了场景路径，则覆盖配置中的场景设置
            if scene_path:
                # 转换为绝对路径
                abs_scene_path = os.path.abspath(scene_path)
                config.habitat.simulator.scene = abs_scene_path
                
                # 更新数据集配置文件
                self._update_dataset_config(abs_scene_path)
            
            # 配置CPU渲染模式，但保持传感器正常工作
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1  # 使用CPU
            config.habitat.simulator.habitat_sim_v0.enable_physics = True
                
            config.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            })
        
        return config
    
    def _update_dataset_config(self, scene_path: str):
        """更新数据集配置文件，将scene_path写入其中"""
        import gzip
        
        dataset_path = "/root/sjm/habitat-lab-old/data/datasets/pointnav/v1/train/train1.json"
        dataset_gz_path = "/root/sjm/habitat-lab-old/data/datasets/pointnav/v1/train/train1.json.gz"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # 读取现有配置或创建新配置
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                dataset_config = json.load(f)
        else:
            dataset_config = {"episodes": []}
        
        # 只更新episodes列表中第一个字典的scene_id字段
        if "episodes" in dataset_config and len(dataset_config["episodes"]) > 0:
            dataset_config["episodes"][0]["scene_id"] = scene_path
        
        # 写回JSON文件
        with open(dataset_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        # 自动生成压缩的.gz文件
        with open(dataset_path, 'rb') as f_in:
            with gzip.open(dataset_gz_path, 'wb') as f_out:
                f_out.write(f_in.read())
    
    def _create_custom_episode(self, scene_path: str, start_position: List[float], target_position: List[float]) -> NavigationEpisode:
        """创建自定义导航episode"""
        episode = NavigationEpisode(
            episode_id="0",
            scene_id=scene_path,
            start_position=start_position,
            start_rotation=[0, 0, 0, 1],  # 默认朝向
            goals=[NavigationGoal(
                position=target_position,
                radius=self.goal_radius
            )]
        )
        return episode
    
    def _calculate_start_position(self, env: habitat.Env, fixed_height: float = 0.41561477) -> List[float]:
        """根据场景边界计算合适的起始位置"""
        lower, upper = env.sim.pathfinder.get_bounds()
        
        # 计算x和z轴的中心位置，y轴使用固定高度
        # 确保所有值都是Python原生float类型，避免numpy类型转换错误
        center_x = float((lower[0] + upper[0]) / 2.0)
        center_z = float((lower[2] + upper[2]) / 2.0)
        start_pos = [center_x, float(fixed_height), center_z]
        
        # 检查计算出的位置是否可导航，如果不可导航则尝试附近的位置
        if not env.sim.pathfinder.is_navigable(start_pos):
            print(f"⚠️  计算的中心位置不可导航: {start_pos}，正在寻找附近的可导航位置...")
            
            # 在中心位置周围搜索可导航的位置
            search_radius = 0.5
            search_step = 0.1
            found_navigable = False
            
            for offset_x in np.arange(-search_radius, search_radius + search_step, search_step):
                for offset_z in np.arange(-search_radius, search_radius + search_step, search_step):
                    test_pos = [float(center_x + offset_x), float(fixed_height), float(center_z + offset_z)]
                    # 确保测试位置在边界内
                    if (lower[0] <= test_pos[0] <= upper[0] and 
                        lower[2] <= test_pos[2] <= upper[2] and
                        env.sim.pathfinder.is_navigable(test_pos)):
                        start_pos = test_pos
                        found_navigable = True
                        print(f"✅ 找到可导航的起始位置: {start_pos}")
                        break
                if found_navigable:
                    break
            
            if not found_navigable:
                print(f"⚠️  未找到可导航的起始位置，使用边界内的默认位置")
                # 使用边界内的一个相对安全的位置
                start_pos = [float(lower[0] + 1.0), float(fixed_height), float(lower[2] + 1.0)]
        else:
            print(f"✅ 计算的中心位置可导航: {start_pos}")
        
        return start_pos
    
    def _validate_and_adjust_goal(self, target_center: np.ndarray, env: habitat.Env) -> np.ndarray:
        """验证并调整目标坐标到导航网格边界内，并确保位置可导航"""
        lower, upper = env.sim.pathfinder.get_bounds()
        print(f"导航网格边界:")
        print(f"  下限: {lower}")
        print(f"  上限: {upper}")

        print(f"目标坐标:",target_center)
        original_target = target_center.copy()
        
        # 首先将坐标调整到边界内
        target_center[0] = np.clip(target_center[0], lower[0], upper[0])
        target_center[1] = np.clip(target_center[1], lower[1], upper[1])
        target_center[2] = np.clip(target_center[2], lower[2], upper[2])
        
        if not np.allclose(original_target, target_center):
            print(f"⚠️  目标坐标已调整到边界内:")
            print(f"  原始坐标: {original_target}")
            print(f"  调整后坐标: {target_center}")
        
        # 检查调整后的位置是否可导航
        target_pos = target_center.tolist()
        if env.sim.pathfinder.is_navigable(target_pos):
            print(f"✅ 目标位置可导航: {target_pos}")
            return target_center
        
        print(f"⚠️  目标位置不可导航: {target_pos}，正在寻找最近的可导航位置...")
        
        # 寻找最近的可导航位置
        best_pos = target_center.copy()
        min_distance = float('inf')
        found_navigable = False
        
        # 搜索参数
        max_search_radius = 2.0  # 最大搜索半径
        search_step = 0.1  # 搜索步长
        
        for radius in np.arange(search_step, max_search_radius + search_step, search_step):
            # 在当前半径的球面上搜索
            for theta in np.arange(0, 2 * np.pi, np.pi / 8):  # 8个方向
                for phi in np.arange(0, np.pi, np.pi / 4):  # 4个高度层
                    # 球坐标转笛卡尔坐标
                    offset_x = radius * np.sin(phi) * np.cos(theta)
                    offset_y = radius * np.cos(phi)
                    offset_z = radius * np.sin(phi) * np.sin(theta)
                    
                    test_pos = [
                        float(target_center[0] + offset_x),
                        float(target_center[1] + offset_y),
                        float(target_center[2] + offset_z)
                    ]
                    
                    # 确保测试位置在边界内
                    if (lower[0] <= test_pos[0] <= upper[0] and 
                        lower[1] <= test_pos[1] <= upper[1] and
                        lower[2] <= test_pos[2] <= upper[2]):
                        
                        if env.sim.pathfinder.is_navigable(test_pos):
                            # 计算与原始目标的距离
                            distance = np.linalg.norm(np.array(test_pos) - original_target)
                            if distance < min_distance:
                                min_distance = distance
                                best_pos = np.array(test_pos)
                                found_navigable = True
            
            # 如果在当前半径找到了可导航位置，就不再扩大搜索范围
            if found_navigable:
                break
        
        if found_navigable:
            print(f"✅ 找到最近的可导航位置: {best_pos.tolist()}")
            print(f"   与原始目标的距离: {min_distance:.3f}")
            return best_pos
        else:
            print(f"⚠️  未找到可导航位置，使用边界调整后的位置: {target_center.tolist()}")
            return target_center
    
    def navigate_to_target_with_mask(self, 
                          ply_path: str, 
                          pred_mask: np.ndarray, 
                          output_path: str,
                          scene_path: Optional[str] = None,
                          video_name: Optional[str] = None) -> Tuple[str, dict]:
        """
        执行导航任务并生成视频（直接使用mask）
        
        Args:
            ply_path: 点云文件路径 (.ply)
            pred_mask: 语义分割掩码数组
            output_path: 输出目录路径
            scene_path: 场景文件路径 (.glb)，如果为None则使用初始化时的scene_path或配置文件中的默认场景
            video_name: 自定义视频名称，如果为None则自动生成
            
        Returns:
            Tuple[str, dict]: (视频文件路径, 导航统计信息)
            
        Raises:
            ValueError: 当未检测到目标点或文件不存在时
            FileNotFoundError: 当输入文件不存在时
        """
        # 验证输入文件
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
        
        # 加载数据
        points = self._load_point_cloud(ply_path)
        
        # 提取目标点
        target_points = points[pred_mask == 1]
        if len(target_points) == 0:
            raise ValueError("未检测到任何目标点")
        
        target_center = target_points.mean(axis=0)
        target_center = self._transform_coordinates(target_center)
        
        # 确定使用的场景路径
        used_scene_path = scene_path or self.scene_path
        if not used_scene_path:
            raise ValueError("必须指定场景文件路径，可以通过scene_path参数或初始化时的scene_path参数指定")
        
        # 验证场景文件是否存在
        if not os.path.exists(used_scene_path):
            raise FileNotFoundError(f"场景文件不存在: {used_scene_path}")
        
        # 创建配置和环境
        config = self._create_habitat_config(used_scene_path)
        
        with habitat.Env(config=config) as env:
            # 如果没有指定起始位置，则根据场景边界计算
            if self.start_position is None:
                calculated_start_position = self._calculate_start_position(env)
            else:
                calculated_start_position = self.start_position.copy()
            
            # 导航统计信息
            nav_stats = {
                "target_points_count": len(target_points),
                "original_target_center": target_center.copy(),
                "start_position": calculated_start_position.copy(),
                "scene_path": used_scene_path
            }
            
            # 验证和调整目标坐标
            target_center = self._validate_and_adjust_goal(target_center, env)
            nav_stats["adjusted_target_center"] = target_center.copy()
            
            # 创建自定义episode
            custom_episode = self._create_custom_episode(
                scene_path=used_scene_path,
                start_position=calculated_start_position,
                target_position=target_center.tolist()
            )
            
            # 手动设置当前episode
            env._current_episode = custom_episode
            
            # episode中的目标位置已经在创建时设置好了
            
            # 创建智能体
            agent = ShortestPathFollowerAgent(
                env=env,
                goal_radius=config.habitat.task.measurements.success.success_distance,
            )
            
            # 执行导航
            observations = env.reset()
            agent.reset()
            
            vis_frames = []
            step_count = 0
            
            # 初始帧
            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames.append(frame)
            
            # 导航循环
            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break
                
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
                step_count += 1
            
            nav_stats["total_steps"] = step_count
            nav_stats["success"] = env.episode_over
            
            # 生成视频
            if video_name is None:
                ply_filename = os.path.splitext(os.path.basename(ply_path))[0]
                scene_id = os.path.splitext(os.path.basename(custom_episode.scene_id))[0]
                video_name = f"{scene_id}_{custom_episode.episode_id}_{ply_filename}"
            
            os.makedirs(output_path, exist_ok=True)
            images_to_video(
                vis_frames, output_path, video_name, 
                fps=self.fps, quality=self.video_quality
            )
            
            # 手动构建视频文件路径，因为images_to_video函数不返回路径
            video_name = video_name.replace(" ", "_").replace("\n", "_")
            video_name_split = video_name.split("/")
            video_name = "/".join(
                video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
            )
            video_path = os.path.join(output_path, video_name)
            
            nav_stats["video_path"] = video_path
            nav_stats["frames_count"] = len(vis_frames)
            
        return video_path, nav_stats

    def navigate_to_target(self, 
                          ply_path: str, 
                          jsonl_path: str, 
                          output_path: str,
                          scene_path: Optional[str] = None,
                          video_name: Optional[str] = None) -> Tuple[str, dict]:
        """
        执行导航任务并生成视频
        
        Args:
            ply_path: 点云文件路径 (.ply)
            jsonl_path: 语义分割结果文件路径 (.jsonl)
            output_path: 输出目录路径
            scene_path: 场景文件路径 (.glb)，如果为None则使用初始化时的scene_path或配置文件中的默认场景
            video_name: 自定义视频名称，如果为None则自动生成
            
        Returns:
            Tuple[str, dict]: (视频文件路径, 导航统计信息)
            
        Raises:
            ValueError: 当未检测到目标点或文件不存在时
            FileNotFoundError: 当输入文件不存在时
        """
        # 验证输入文件
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL文件不存在: {jsonl_path}")
        
        # 加载数据
        points = self._load_point_cloud(ply_path)
        pred_mask = self._load_segmentation_mask(jsonl_path)
        
        # 提取目标点
        target_points = points[pred_mask == 1]
        if len(target_points) == 0:
            raise ValueError("未检测到任何目标点")
        
        target_center = target_points.mean(axis=0)
        target_center = self._transform_coordinates(target_center)
        
        # 确定使用的场景路径
        used_scene_path = scene_path or self.scene_path
        if not used_scene_path:
            raise ValueError("必须指定场景文件路径，可以通过scene_path参数或初始化时的scene_path参数指定")
        
        # 验证场景文件是否存在
        if not os.path.exists(used_scene_path):
            raise FileNotFoundError(f"场景文件不存在: {used_scene_path}")
        
        # 创建配置和环境
        config = self._create_habitat_config(used_scene_path)
        
        with habitat.Env(config=config) as env:
            # 如果没有指定起始位置，则根据场景边界计算
            if self.start_position is None:
                calculated_start_position = self._calculate_start_position(env)
            else:
                calculated_start_position = self.start_position.copy()
            
            # 导航统计信息
            nav_stats = {
                "target_points_count": len(target_points),
                "original_target_center": target_center.copy(),
                "start_position": calculated_start_position.copy(),
                "scene_path": used_scene_path
            }
            
            # 验证和调整目标坐标
            target_center = self._validate_and_adjust_goal(target_center, env)
            nav_stats["adjusted_target_center"] = target_center.copy()
            
            # 创建自定义episode
            custom_episode = self._create_custom_episode(
                scene_path=used_scene_path,
                start_position=calculated_start_position,
                target_position=target_center.tolist()
            )
            
            # 手动设置当前episode
            env._current_episode = custom_episode
            
            # episode中的目标位置已经在创建时设置好了
            
            # 创建智能体
            agent = ShortestPathFollowerAgent(
                env=env,
                goal_radius=config.habitat.task.measurements.success.success_distance,
            )
            
            # 执行导航
            observations = env.reset()
            agent.reset()
            
            vis_frames = []
            step_count = 0
            
            # 初始帧
            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames.append(frame)
            
            # 导航循环
            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break
                
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
                step_count += 1
            
            nav_stats["total_steps"] = step_count
            nav_stats["success"] = env.episode_over
            
            # 生成视频
            if video_name is None:
                ply_filename = os.path.splitext(os.path.basename(ply_path))[0]
                scene_id = os.path.splitext(os.path.basename(custom_episode.scene_id))[0]
                video_name = f"{scene_id}_{custom_episode.episode_id}_{ply_filename}"
            
            os.makedirs(output_path, exist_ok=True)
            video_path = images_to_video(
                vis_frames, output_path, video_name, 
                fps=self.fps, quality=self.video_quality
            )
            
            nav_stats["video_path"] = video_path
            nav_stats["frames_count"] = len(vis_frames)
            
        return video_path, nav_stats
    
    def batch_navigate(self, 
                      tasks: List[Tuple[str, str, str, Optional[str]]], 
                      output_base_path: str) -> List[Tuple[str, dict]]:
        """
        批量执行导航任务
        
        Args:
            tasks: 任务列表，每个任务为 (ply_path, jsonl_path, task_name, scene_path)
            output_base_path: 输出基础路径
            
        Returns:
            List[Tuple[str, dict]]: 每个任务的结果列表
        """
        results = []
        
        for i, task_info in enumerate(tasks):
            try:
                if len(task_info) == 3:
                    ply_path, jsonl_path, task_name = task_info
                    scene_path = None
                else:
                    ply_path, jsonl_path, task_name, scene_path = task_info
                    
                output_path = os.path.join(output_base_path, f"task_{i+1}_{task_name}")
                video_path, stats = self.navigate_to_target(
                    ply_path, jsonl_path, output_path, scene_path, task_name
                )
                results.append((video_path, stats))
                print(f"✅ 任务 {i+1}/{len(tasks)} 完成: {task_name}")
            except Exception as e:
                print(f"❌ 任务 {i+1}/{len(tasks)} 失败: {task_name}, 错误: {str(e)}")
                results.append((None, {"error": str(e)}))
        
        return results


# 便捷函数
def quick_navigate(ply_path: str, 
                  jsonl_path: str, 
                  output_path: str,
                  scene_path: Optional[str] = None,
                  yaml_path: str = "/root/sjm/habitat-lab-old/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_scannet.yaml") -> str:
    """
    快速导航函数
    
    Args:
        ply_path: 点云文件路径
        jsonl_path: 语义分割结果文件路径
        output_path: 输出路径
        scene_path: 场景文件路径 (.glb)
        yaml_path: Habitat配置文件路径
        
    Returns:
        str: 生成的视频文件路径
    """
    api = NavigationAPI(yaml_path=yaml_path, scene_path=scene_path)
    video_path, stats = api.navigate_to_target(ply_path, jsonl_path, output_path, scene_path)
    print(f"导航完成! 视频保存至: {video_path}")
    print(f"导航统计: {stats}")
    return video_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Habitat导航API")
    parser.add_argument("--ply_path", required=True, help="点云文件路径")
    parser.add_argument("--jsonl_path", required=True, help="语义分割结果文件路径")
    parser.add_argument("--output_path", required=True, help="输出目录路径")
    parser.add_argument("--scene_path", help="场景文件路径 (.glb)")
    parser.add_argument("--yaml_path", 
                       default="/root/sjm/habitat-lab-old/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_scannet.yaml",
                       help="Habitat配置文件路径")
    parser.add_argument("--video_name", help="自定义视频名称")
    
    args = parser.parse_args()
    
    try:
        api = NavigationAPI(yaml_path=args.yaml_path, scene_path=args.scene_path)
        video_path, stats = api.navigate_to_target(
            args.ply_path, 
            args.jsonl_path, 
            args.output_path,
            args.scene_path,
            args.video_name
        )
        print(f"✅ 导航成功完成!")
        print(f"📹 视频文件: {video_path}")
        print(f"📊 导航统计: {stats}")
    except Exception as e:
        print(f"❌ 导航失败: {str(e)}")