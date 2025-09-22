import torch
import numpy as np
import open3d as o3d
import argparse
import os
import glob


def convert_single_file(pth_path, ply_path, data_type, vert_key, color_key, face_key):
    """
    将单个 .pth 文件转换为 .ply 文件。
    此版本经过修改，可以处理坐标和颜色在字典中作为独立键的情况。
    """
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"    - Error loading .pth file: {e}")
        raise

    if data_type == 'pointcloud':
        pcd = o3d.geometry.PointCloud()

        # --- 新的逻辑：处理字典格式的点云 ---
        if isinstance(data, dict):
            # 1. 提取坐标 (必要)
            if vert_key not in data:
                raise KeyError(f"坐标键 '{vert_key}' 不在字典中。可用键: {list(data.keys())}")

            points = data[vert_key]
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()

            pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
            print(f"    - 从键 '{vert_key}' 加载了 {len(points)} 个点。")

            # 2. 提取颜色 (可选)
            if color_key in data:
                colors = data[color_key]
                if isinstance(colors, torch.Tensor):
                    colors = colors.cpu().numpy()

                if colors.shape[0] != points.shape[0]:
                    print(
                        f"    - 警告: 点的数量 ({points.shape[0]}) 和颜色的数量 ({colors.shape[0]}) 不匹配。将忽略颜色。")
                else:
                    # 如果颜色值范围是 0-255，归一化到 0-1
                    if np.max(colors) > 1.0:
                        colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64))
                    print(f"    - 从键 '{color_key}' 加载了颜色。")
            else:
                print(f"    - 警告: 未在字典中找到颜色键 '{color_key}'。")

        # --- 旧的逻辑：处理原始张量格式的点云 (作为备用) ---
        elif isinstance(data, torch.Tensor):
            points = data.cpu().numpy()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
            if points.shape[1] >= 6:
                colors = points[:, 3:6]
                if np.max(colors) > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}。数据应为字典或张量。")

        o3d.io.write_point_cloud(ply_path, pcd)

    elif data_type == 'mesh':
        # 网格处理逻辑保持不变
        if not isinstance(data, dict):
            raise ValueError("对于网格类型, 数据必须是字典。")
        if vert_key not in data or face_key not in data:
            raise KeyError(f"网格所需的键 ('{vert_key}', '{face_key}') 不在字典中。可用键: {list(data.keys())}")

        vertices = data[vert_key].cpu().numpy()
        faces = data[face_key].cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(ply_path, mesh)
    else:
        raise ValueError(f"未知的数据类型 '{data_type}'。请使用 'pointcloud' 或 'mesh'。")


def batch_convert(input_dir, output_dir, data_type, vert_key, color_key, face_key):
    if not os.path.isdir(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在。")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件将保存至: '{output_dir}'")

    pth_files = glob.glob(os.path.join(input_dir, '*.pth'))
    if not pth_files:
        print(f"在 '{input_dir}' 中未找到 .pth 文件。")
        return

    print(f"找到 {len(pth_files)} 个 .pth 文件，开始转换...")
    success_count, fail_count = 0, 0

    for pth_path in pth_files:
        base_filename = os.path.basename(pth_path)
        ply_filename = os.path.splitext(base_filename)[0] + '.ply'
        ply_path = os.path.join(output_dir, ply_filename)

        print(f"\n[处理中] '{base_filename}'")
        try:
            convert_single_file(pth_path, ply_path, data_type, vert_key, color_key, face_key)
            print(f"  -> [成功] 已保存为 '{ply_path}'")
            success_count += 1
        except Exception as e:
            print(f"  -> [失败] 转换时发生错误: {e}")
            fail_count += 1

    print("\n" + "=" * 50)
    print("批量转换完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将一个文件夹内的 .pth 3D文件转换为 .ply 文件。")
    parser.add_argument("input_dir", type=str, help="包含 .pth 文件的输入文件夹路径。")
    parser.add_argument("output_dir", type=str, help="用于保存 .ply 文件的输出文件夹路径。")
    parser.add_argument("--type", type=str, required=True, choices=['pointcloud', 'mesh'],
                        help="所有.pth文件中的3D数据类型: 'pointcloud' (点云) 或 'mesh' (网格)。")
    # 修改了默认值以匹配您的数据格式
    parser.add_argument("--vert_key", type=str, default="coord",
                        help="顶点/点云坐标数据的字典键名。默认: 'coord'")
    parser.add_argument("--color_key", type=str, default="color",
                        help="颜色数据的字典键名 (可选)。默认: 'color'")
    parser.add_argument("--face_key", type=str, default="faces",
                        help="面数据的字典键名 (仅用于 'mesh' 类型)。默认: 'faces'")

    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir, args.type, args.vert_key, args.color_key, args.face_key)