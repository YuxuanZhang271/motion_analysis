import json
import numpy as np
import open3d as o3d
import smplx
import torch


def load_data(path: str) -> dict: 
    with open(path, "r") as f: 
        data = json.load(f)
    return data


def load_smplx(device: str) -> smplx.SMPLX: 
    model = smplx.create(
        model_path             = "model",
        model_type             = "smplx",
        gender                 = "neutral",
        num_betas              = 10,
        use_pca                = False, 
        flat_hand_mean         = True,
        create_global_orient   = True,
        create_body_pose       = True,
        create_left_hand_pose  = True,
        create_right_hand_pose = True,
        create_betas           = True,
        create_expression      = False,
        create_jaw_pose        = False,
        create_leye_pose       = False,
        create_reye_pose       = False
    ).to(device)
    model.eval()
    return model


def list_to_tensor(data, i, device): 
    t   = torch.as_tensor(data["transl"]         [i]).to(device=device, dtype=torch.float32)
    go  = torch.as_tensor(data["global_orient"]  [i]).to(device=device, dtype=torch.float32)
    bp  = torch.as_tensor(data["body_pose"]      [i]).to(device=device, dtype=torch.float32)
    b   = torch.as_tensor(data["betas"]          [i]).to(device=device, dtype=torch.float32)
    lhp = torch.as_tensor(data["left_hand_pose"] [i]).to(device=device, dtype=torch.float32)
    rhp = torch.as_tensor(data["right_hand_pose"][i]).to(device=device, dtype=torch.float32)
    return t, go, bp, b, lhp, rhp


def param_to_mesh(
    model: smplx.SMPLX, 
    transl: torch.Tensor, 
    global_orient: torch.Tensor, 
    body_pose: torch.Tensor, 
    betas: torch.Tensor, 
    left_hand_pose: torch.Tensor, 
    right_hand_pose: torch.Tensor
) -> o3d.geometry.TriangleMesh: 
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
        faces    = model.faces.astype(np.int32)

    mesh           = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.7, 0.75, 0.9))
    
    return mesh


def visualize(data, device): 
    model = load_smplx(device)

    seq_len = data["body_pose"].shape[0]
    for i in range(seq_len): 
        transl, global_orient, body_pose, betas, left_hand_pose, right_hand_pose = list_to_tensor(data, i, device)

        mesh = param_to_mesh(model, transl, global_orient, body_pose, betas, left_hand_pose, right_hand_pose)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SMPL-X", width=960, height=720, visible=True)
        vis.add_geometry(mesh)

        break


def _test(): 
    data = load_data("dataset/train/dumbbell_hammer_curls_s03.json")
    # print(list(data.keys()))
    if torch.backends.mps.is_available(): 
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu" 
    visualize(data, device)


if __name__ == "__main__": 
    _test()