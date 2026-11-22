"""Build InterAct-style BEHAVE and OMOMO sequences from the local raw fits.

The geometric steps follow external/InterAct/process/process_behave.py,
process_omomo.py, canonicalize_obj.py, and canonicalize_human.py. Results are
written under datasets/interact and the files in datasets/behave and
datasets/OMOMO are left unchanged.

OMOMO train and test pickles both use integer keys starting at 0. InterAct's
script updates one dict with the other and drops 602 training sequences. This
runner keeps every sequence and records the split in split.json.

Text tables from the InterAct release are not in this checkout, so each full
sequence is one segment. Hand-contact optimization is a separate optional pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_BEHAVE = ROOT / "datasets" / "behave" / "behave-30fps-params-v1"
RAW_BEHAVE_OBJECTS = ROOT / "datasets" / "behave" / "objects" / "objects"
RAW_OMOMO = ROOT / "datasets" / "OMOMO" / "OMOMO" / "data"
MODELS = ROOT / "datasets" / "smplx_models"
OUT = ROOT / "datasets" / "interact"


def _quat_normalize(x, eps=1e-8):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + eps)


def _quat_mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
    ], axis=-1)


def _quat_mul_vec(q, x):
    t = 2.0 * np.cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)


def _quat_inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q


def _quat_between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] + np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y),
    ], axis=-1)


def _quat_fk(lrot, lpos, parents):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(_quat_mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(_quat_mul(gr[parents[i]], lrot[..., i:i + 1, :]))
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def _quat_ik(grot, gpos, parents):
    return (
        np.concatenate([grot[..., :1, :], _quat_mul(_quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :])], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            _quat_mul_vec(_quat_inv(grot[..., parents[1:], :]), gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2),
    )


def _rotate_at_frame(X, Q, obj_x, obj_q, trans2joint, parents):
    """InterAct process_omomo.rotate_at_frame_w_obj with floor_z=True."""
    global_q, global_x = _quat_fk(Q, X, parents)
    key = global_q[:, :1, 0:1, :]
    forward = np.array([1, 1, 0], dtype=np.float32)[None, None, None, :] * _quat_mul_vec(
        key, np.array([1, 0, 0], dtype=np.float32)[None, None, None, :])
    forward = _quat_normalize(forward)
    yrot = _quat_normalize(_quat_between(np.array([1, 0, 0], dtype=np.float32), forward))
    new_glob_Q = _quat_mul(_quat_inv(yrot), global_q)
    new_glob_X = _quat_mul_vec(_quat_inv(yrot), global_x)
    new_obj_q = _quat_mul(_quat_inv(yrot[:, 0, :, :]), obj_q)
    obj_trans = obj_x + trans2joint[:, None, :]
    obj_trans = _quat_mul_vec(_quat_inv(yrot[:, 0, :, :]), obj_trans) - trans2joint[:, None, :]
    local_q, local_x = _quat_ik(new_glob_Q, new_glob_X, parents)
    return local_x, local_q, obj_trans, new_obj_q


def _aa_to_quat_wxyz(aa):
    shape = aa.shape[:-1]
    xyzw = Rotation.from_rotvec(np.asarray(aa, np.float64).reshape(-1, 3)).as_quat()
    return xyzw[:, [3, 0, 1, 2]].reshape(*shape, 4).astype(np.float32)


def _quat_wxyz_to_matrix(q):
    shape = q.shape[:-1]
    xyzw = np.asarray(q, np.float64).reshape(-1, 4)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix().reshape(*shape, 3, 3).astype(np.float32)


def _matrix_to_aa(mat):
    shape = mat.shape[:-2]
    return Rotation.from_matrix(np.asarray(mat, np.float64).reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3).astype(np.float32)


def _parents():
    with np.load(MODELS / "smplx" / "SMPLX_MALE.npz") as archive:
        parents = np.array(archive["kintree_table"][0, :22], dtype=np.int64)
    parents[0] = -1
    return parents


def _load_models():
    import smplx
    behave = {
        "male": smplx.create(MODELS, model_type="smplh", gender="male", use_pca=False, ext="pkl", num_betas=10),
        "female": smplx.create(MODELS, model_type="smplh", gender="female", use_pca=False, ext="pkl", num_betas=10),
    }
    omomo = {
        gender: smplx.create(
            MODELS, model_type="smplx", gender=gender, num_betas=16, use_pca=False,
            flat_hand_mean=True, ext="npz")
        for gender in ("male", "female")
    }
    return behave, omomo


def _forward(model, poses, betas, trans, *, kind, need_vertices=False, vertex_frames=None, chunk=256):
    """Pelvis trajectory, and optionally the minimum vertex height over selected frames."""
    poses = np.asarray(poses, np.float32)
    trans = np.asarray(trans, np.float32)
    beta = np.asarray(betas, np.float32).reshape(-1)
    count = 10 if kind == "smplh" else 16
    beta_t = torch.from_numpy(beta[:count])
    pelvis = np.empty((len(poses), 3), np.float32)
    min_y = np.inf
    with torch.inference_mode():
        for start in range(0, len(poses), chunk):
            end = min(start + chunk, len(poses))
            take = end - start
            kwargs = dict(
                body_pose=torch.from_numpy(poses[start:end, 3:66]),
                global_orient=torch.from_numpy(poses[start:end, :3]),
                left_hand_pose=torch.from_numpy(poses[start:end, 66:111]),
                right_hand_pose=torch.from_numpy(poses[start:end, 111:156]),
                betas=beta_t.expand(take, -1),
                transl=torch.from_numpy(trans[start:end]),
            )
            if kind == "smplx":
                zeros = torch.zeros(take, 3)
                kwargs.update(jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros.clone(),
                              expression=torch.zeros(take, 10))
            output = model(**kwargs)
            pelvis[start:end] = output.joints[:, 0].numpy()
            if need_vertices and vertex_frames is not None and start < vertex_frames:
                used = min(end, vertex_frames) - start
                min_y = min(min_y, float(output.vertices[:used, :, 1].min()))
            elif need_vertices and vertex_frames is None:
                min_y = min(min_y, float(output.vertices[:, :, 1].min()))
    return pelvis, min_y


def _object_min_y(vertices, angles, trans, frames):
    frames = min(frames, len(angles), len(trans))
    if frames < 1:
        return np.inf
    mats = Rotation.from_rotvec(angles[:frames]).as_matrix()
    points = vertices[None] @ np.transpose(mats, (0, 2, 1)) + trans[:frames, None, :]
    return float(points[..., 1].min())


def _apply_x_rotation(poses, trans, obj_rotvec_or_matrix, obj_trans, pelvis, degrees_rad):
    rotation = Rotation.from_euler("x", degrees_rad)
    poses = np.array(poses, np.float32, copy=True)
    poses[:, :3] = (rotation * Rotation.from_rotvec(poses[:, :3])).as_rotvec().astype(np.float32)
    trans = rotation.apply(trans).astype(np.float32)
    if obj_rotvec_or_matrix.ndim == 3:
        obj_rotation = Rotation.from_matrix(obj_rotvec_or_matrix)
    else:
        obj_rotation = Rotation.from_rotvec(obj_rotvec_or_matrix)
    obj_angles = (rotation * obj_rotation).as_rotvec().astype(np.float32)
    obj_delta = rotation.apply(obj_trans - pelvis).astype(np.float32)
    return poses, trans, obj_angles, obj_delta


def _ground(model, kind, poses, betas, trans, obj_angles, obj_delta, name, mesh, frames):
    poses, trans = np.array(poses, copy=True), np.array(trans, copy=True)
    pelvis, human_y = _forward(model, poses, betas, trans, kind=kind, need_vertices=True, vertex_frames=frames)
    obj_trans = pelvis + obj_delta
    object_y = _object_min_y(mesh, obj_angles, obj_trans, frames if frames is not None else len(obj_trans))
    drop = min(human_y, object_y)
    obj_trans = np.array(obj_trans, copy=True)
    obj_trans[:, 1] -= drop
    trans[:, 1] -= drop
    return {
        "poses": poses.astype(np.float32), "betas": np.asarray(betas, np.float32).reshape(-1),
        "trans": trans.astype(np.float32), "gender": name,
    }, {"angles": obj_angles.astype(np.float32), "trans": obj_trans.astype(np.float32)}


def _heading_rotation(poses):
    matrix = Rotation.from_rotvec(poses[0, :3]).as_matrix()
    norm = np.sqrt(matrix[0, 0] ** 2 + matrix[2, 0] ** 2)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    cos, sin = matrix[0, 0] / norm, matrix[2, 0] / norm
    rotation_v = np.eye(3, dtype=np.float32)
    rotation_v[0, 0] = cos
    rotation_v[0, 2] = -sin
    rotation_v[2, 0] = sin
    rotation_v[2, 2] = cos
    return np.linalg.inv(rotation_v).astype(np.float32)


def _canonicalize_human(model, kind, human, obj, mesh):
    poses = np.array(human["poses"], np.float32, copy=True)
    trans = np.array(human["trans"], np.float32, copy=True)
    betas = human["betas"]
    pelvis, _ = _forward(model, poses, betas, trans, kind=kind)
    centroid = pelvis[0].copy()
    rotation = _heading_rotation(poses)
    rotator = Rotation.from_matrix(rotation)
    trans = trans - centroid
    pelvis = pelvis - centroid
    pelvis_original = pelvis - trans
    trans = ((trans + pelvis_original) @ rotation.T - pelvis_original).astype(np.float32)
    poses[:, :3] = (rotator * Rotation.from_rotvec(poses[:, :3])).as_rotvec().astype(np.float32)
    obj_angles = (rotator * Rotation.from_rotvec(obj["angles"])).as_rotvec().astype(np.float32)
    obj_trans = ((obj["trans"] - centroid) @ rotation.T).astype(np.float32)
    _, human_y = _forward(model, poses, betas, trans, kind=kind, need_vertices=True, vertex_frames=30)
    object_y = _object_min_y(mesh, obj_angles, obj_trans, 30)
    drop = min(human_y, object_y)
    trans = np.array(trans, copy=True)
    obj_trans = np.array(obj_trans, copy=True)
    trans[:, 1] -= drop
    obj_trans[:, 1] -= drop
    human = {"poses": poses, "betas": np.asarray(betas, np.float32).reshape(-1), "trans": trans, "gender": human["gender"]}
    obj = {"angles": obj_angles, "trans": obj_trans, "name": obj["name"]}
    return human, obj


def _save_pair(directory, human, obj):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "human.npz", **human)
    np.savez(directory / "object.npz", **obj)


def _read_obj(path):
    """Read vertices and triangular faces without loading textures."""
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(item) for item in line.split()[1:4]])
        elif line.startswith("f "):
            corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for index in range(1, len(corners) - 1):
                faces.append([corners[0], corners[index], corners[index + 1]])
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)


def _write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def prepare_behave(models, limit):
    behave_out = OUT / "behave"
    objects = behave_out / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    meshes = {}
    names = sorted(p.name for p in RAW_BEHAVE.iterdir() if p.is_dir())
    if limit:
        names = names[:limit]
    done = 0
    for name in tqdm(names, desc="behave"):
        canonical = behave_out / "sequences_canonical" / name
        if (canonical / "human.npz").exists() and (canonical / "object.npz").exists():
            done += 1
            continue
        info = json.loads((RAW_BEHAVE / name / "info.json").read_text(encoding="utf-8"))
        gender = str(info["gender"])
        obj_name = str(info["cat"])
        destination = objects / obj_name / f"{obj_name}.obj"
        if obj_name not in meshes:
            if not destination.exists():
                vertices, faces = _read_obj(RAW_BEHAVE_OBJECTS / obj_name / f"{obj_name}.obj")
                vertices = vertices - vertices.mean(axis=0, keepdims=True)
                _write_obj(destination, vertices, faces)
            else:
                vertices, _ = _read_obj(destination)
            meshes[obj_name] = vertices
        with np.load(RAW_BEHAVE / name / "smpl_fit_all.npz") as archive:
            poses, betas, trans = archive["poses"].copy(), archive["betas"][0].copy(), archive["trans"].copy()
        with np.load(RAW_BEHAVE / name / "object_fit_all.npz") as archive:
            obj_angles, obj_trans = archive["angles"].copy(), archive["trans"].copy()
        pelvis, _ = _forward(models[gender], poses, betas, trans, kind="smplh")
        poses, trans, obj_angles, obj_delta = _apply_x_rotation(poses, trans, obj_angles, obj_trans, pelvis, -np.pi)
        human, obj = _ground(models[gender], "smplh", poses, betas, trans, obj_angles, obj_delta, gender, meshes[obj_name], 30)
        obj["name"] = obj_name
        _save_pair(behave_out / "sequences" / name, human, obj)
        _save_pair(behave_out / "sequences_seg" / name, human, obj)
        human, obj = _canonicalize_human(models[gender], "smplh", human, obj, meshes[obj_name])
        _save_pair(canonical, human, obj)
        done += 1
    print(f"BEHAVE canonical sequences: {done}", flush=True)


def _omomo_records():
    import joblib
    records = []
    for split, filename in (("train", "train_diffusion_manip_seq_joints24.p"), ("test", "test_diffusion_manip_seq_joints24.p")):
        table = joblib.load(RAW_OMOMO / filename)
        for record in table.values():
            records.append((split, record))
    return records


def _stage_omomo_objects(records):
    """Scale meshes, remember the pre-centering vertices, and write the centered mesh."""
    objects = OUT / "omomo" / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    scale_for = {}
    for _, record in records:
        obj_name = str(record["seq_name"]).split("_")[1]
        scale_for.setdefault(obj_name, float(np.asarray(record["obj_scale"]).reshape(-1)[0]))
    raw, centers, centered = {}, {}, {}
    for obj_name, scale in scale_for.items():
        destination = objects / obj_name / f"{obj_name}.obj"
        center_path = objects / obj_name / "center.npy"
        source = RAW_OMOMO / "captured_objects" / f"{obj_name}_cleaned_simplified.obj"
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and center_path.exists():
            center = np.load(center_path).astype(np.float32)
            centered_vertices, _ = _read_obj(destination)
            vertices = centered_vertices + center
        else:
            vertices, faces = _read_obj(source)
            vertices = vertices * np.float32(scale)
            center = vertices.mean(axis=0).astype(np.float32)
            _write_obj(destination, vertices - center, faces)
            np.save(center_path, center)
            centered_vertices = vertices - center
        raw[obj_name] = vertices
        centers[obj_name] = center
        centered[obj_name] = centered_vertices.astype(np.float32)
    return raw, centers, centered


def prepare_omomo(models, parents, limit):
    records = _omomo_records()
    raw_meshes, centers, centered_meshes = _stage_omomo_objects(records)
    if limit:
        records = records[:limit]
    omomo_out = OUT / "omomo"
    splits = {}
    if (omomo_out / "split.json").exists():
        splits.update(json.loads((omomo_out / "split.json").read_text(encoding="utf-8")))
    done = 0
    for split, record in tqdm(records, desc="omomo"):
        seq_name = str(record["seq_name"])
        obj_name = seq_name.split("_")[1]
        splits[seq_name] = split
        canonical = omomo_out / "sequences_canonical" / seq_name
        if (canonical / "human.npz").exists() and (canonical / "object.npz").exists():
            done += 1
            continue
        gender = str(record["gender"])
        betas = np.asarray(record["betas"], np.float32).reshape(-1)
        trans = np.asarray(record["trans"], np.float32)
        root = np.asarray(record["root_orient"], np.float32)
        body = np.asarray(record["pose_body"], np.float32).reshape(len(trans), 21, 3)
        obj_trans = np.asarray(record["obj_trans"], np.float32)[:, :, 0]
        obj_rot = np.asarray(record["obj_rot"], np.float32)
        rest = np.asarray(record["rest_offsets"], np.float32)
        trans2joint = np.asarray(record["trans2joint"], np.float32)
        poses_aa = np.concatenate([root[:, None, :], body], axis=1)
        positions = np.repeat(rest[None], len(trans), axis=0)
        positions[:, 0, :] = trans
        quats = _aa_to_quat_wxyz(poses_aa)
        obj_q = _aa_to_quat_wxyz(_matrix_to_aa(obj_rot))
        _, _, new_obj_x, new_obj_q = _rotate_at_frame(
            positions[None], quats[None], obj_trans[None], obj_q[None], trans2joint[None], parents)
        positions, quats, _, _ = _rotate_at_frame(
            positions[None], quats[None], np.asarray(record["obj_com_pos"], np.float32)[None], obj_q[None],
            trans2joint[None], parents)
        new_aa = _matrix_to_aa(_quat_wxyz_to_matrix(quats[0]))
        poses = np.concatenate([
            new_aa[:, 0], new_aa[:, 1:].reshape(len(trans), -1), np.zeros((len(trans), 90), np.float32),
        ], axis=1)
        human = {"poses": poses, "betas": betas, "trans": positions[0, :, 0].astype(np.float32), "gender": gender}
        obj_rot_out = _quat_wxyz_to_matrix(new_obj_q[0])
        obj = {"rot": obj_rot_out, "trans": new_obj_x[0].astype(np.float32), "name": obj_name}
        pelvis, _ = _forward(models[gender], human["poses"], betas, human["trans"], kind="smplx")
        poses, trans, obj_angles, obj_delta = _apply_x_rotation(
            human["poses"], human["trans"], obj["rot"], obj["trans"], pelvis, -np.pi / 2)
        human, obj = _ground(
            models[gender], "smplx", poses, betas, trans, obj_angles, obj_delta, gender, raw_meshes[obj_name], None)
        obj["name"] = obj_name
        obj["trans"] = obj["trans"] + Rotation.from_rotvec(obj["angles"]).apply(centers[obj_name]).astype(np.float32)
        _save_pair(omomo_out / "sequences" / seq_name, human, obj)
        _save_pair(omomo_out / "sequences_seg" / seq_name, human, obj)
        human, obj = _canonicalize_human(models[gender], "smplx", human, obj, centered_meshes[obj_name])
        _save_pair(canonical, human, obj)
        done += 1
        if done % 200 == 0:
            (omomo_out / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (omomo_out / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"OMOMO canonical sequences: {done}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("behave", "omomo", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N sequences of each selected dataset")
    args = parser.parse_args()
    torch.set_num_threads(8)
    behave_models, omomo_models = _load_models()
    limit = args.limit or None
    if args.dataset in ("behave", "all"):
        prepare_behave(behave_models, limit)
    if args.dataset in ("omomo", "all"):
        prepare_omomo(omomo_models, _parents(), limit)


if __name__ == "__main__":
    main()
