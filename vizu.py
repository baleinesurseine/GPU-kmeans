import open3d as o3d
import argparse

parser = argparse.ArgumentParser(
        description='Cluster points cloud.')
parser.add_argument("file", help="filename")
args = parser.parse_args()

cloud = o3d.io.read_point_cloud(args.file)
o3d.visualization.draw_geometries_with_custom_animation([cloud])

