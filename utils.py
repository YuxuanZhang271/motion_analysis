import json
import numpy as np
import open3d as o3d
import smplx
import torch


def load_data(path): 
    with open(path, "r") as f: 
        data = json.load(f)
    return data


def visualize(data, device): 
    model = smplx.create(
        model_path="model",
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        use_pca=False, 
        flat_hand_mean=True,
        create_global_orient=True,
        create_body_pose=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_betas=True,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False
    ).to(device)
    model.eval()

    transl          = data["transl"]
    global_orient   = data["global_orient"]
    body_pose       = data["body_pose"]
    betas           = data["betas"]
    left_hand_pose  = data["left_hand_pose"]
    right_hand_pose = data["right_hand_pose"]

    seq_len = body_pose.shape[0]
    for i in range(seq_len): 
        t   = torch.as_tensor(transl[i]).to(device=device, dtype=torch.float32)
        go  = torch.as_tensor(global_orient[i]).to(device=device, dtype=torch.float32)
        bp  = torch.as_tensor(body_pose[i]).to(device=device, dtype=torch.float32)
        b   = torch.as_tensor(betas[i]).to(device=device, dtype=torch.float32)
        lhp = torch.as_tensor(left_hand_pose[i]).to(device=device, dtype=torch.float32)
        rhp = torch.as_tensor(right_hand_pose[i]).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(
                transl          = transl,
                global_orient   = global_orient,
                body_pose       = body_pose,
                betas           = betas,
                left_hand_pose  = left_hand_pose,
                right_hand_pose = right_hand_pose,
                return_verts    = True
            )
            vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
            faces = model.faces.astype(np.int32)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color((0.7, 0.75, 0.9))

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="SMPL-X", width=960, height=720, visible=True)
            vis.add_geometry(mesh)

            ctr = vis.get_view_control()
            ctr.set_front(np.array(front))
            ctr.set_lookat(np.array(lookat))
            ctr.set_up(np.array(up))
            ctr.set_zoom(1.0)


def _test(): 
    data = load_data("dataset/train/dumbbell_hammer_curls_s03.json")
    # print(list(data.keys()))
    if torch.backends.mps.is_available(): 
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu" 


if __name__ == "__main__": 
    _test()