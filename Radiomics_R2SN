import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def make_extractor(bin_width=25.0, normalize=False, include_shape=True):
    settings = {"binWidth": bin_width}
    if normalize:
        settings["normalize"] = True

    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    ext.disableAllFeatures()

    classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    if include_shape:
        classes.append("shape")

    for c in classes:
        ext.enableFeatureClassByName(c)

    return ext


def clean_features(feats):
    out = {}
    for k, v in feats.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def same_space(a, b, tol=1e-6):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def get_binary_mask_from_label(label_img, label):
    mask = sitk.Equal(label_img, int(label))
    return sitk.Cast(mask, sitk.sitkUInt8)


def extract_global_radiomics(image, global_mask, out_csv, bin_width=25.0, normalize=False):
    ext = make_extractor(bin_width=bin_width, normalize=normalize, include_shape=True)
    feats = clean_features(ext.execute(image, global_mask))
    df = pd.DataFrame([feats])
    df.to_csv(out_csv, index=False)
    return df


def extract_regional_features(image, region_mask, bin_width=25.0, normalize=False, include_shape=False):
    ext = make_extractor(bin_width=bin_width, normalize=normalize, include_shape=include_shape)

    labels = np.unique(sitk.GetArrayViewFromImage(region_mask))
    labels = [int(x) for x in labels if x > 0]

    rows = []
    for label in labels:
        roi = get_binary_mask_from_label(region_mask, label)
        try:
            feats = clean_features(ext.execute(image, roi))
            feats["region_id"] = label
            rows.append(feats)
        except Exception:
            continue

    if not rows:
        raise RuntimeError("No valid regional features were extracted.")

    df = pd.DataFrame(rows).set_index("region_id").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    return df


def build_rrsn(regional_df, out_csv):
    x = regional_df.copy()

    # Z-score by feature
    x = (x - x.mean(axis=0)) / (x.std(axis=0, ddof=0) + 1e-12)
    x = x.fillna(0.0)

    if len(x) == 1:
        net = np.array([[1.0]])
    else:
        net = np.corrcoef(x.to_numpy(), rowvar=True)
        net = np.nan_to_num(net, nan=0.0)
        np.fill_diagonal(net, 1.0)

    net_df = pd.DataFrame(net, index=x.index, columns=x.index)
    net_df.index.name = "region_id"
    net_df.to_csv(out_csv)
    return net_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--region_mask", required=True, help="Region label mask path")
    parser.add_argument("--global_mask", default=None, help="Global ROI mask path")
    parser.add_argument("--out_radiomics", default="radiomics.csv", help="Output radiomics CSV")
    parser.add_argument("--out_rrsn", default="rrsn.csv", help="Output RRSN CSV")
    parser.add_argument("--bin_width", type=float, default=25.0, help="PyRadiomics binWidth")
    parser.add_argument("--normalize", action="store_true", help="Enable intensity normalization")
    parser.add_argument(
        "--include_shape_in_network",
        action="store_true",
        help="Include shape features in regional network",
    )
    args = parser.parse_args()

    image = sitk.ReadImage(args.image)
    region_mask = sitk.ReadImage(args.region_mask)

    if args.global_mask is None:
        global_mask = sitk.Cast(region_mask > 0, sitk.sitkUInt8)
    else:
        global_mask = sitk.ReadImage(args.global_mask)

    if not same_space(image, region_mask):
        raise ValueError("image and region_mask are not in the same space")
    if not same_space(image, global_mask):
        raise ValueError("image and global_mask are not in the same space")

    extract_global_radiomics(
        image=image,
        global_mask=global_mask,
        out_csv=args.out_radiomics,
        bin_width=args.bin_width,
        normalize=args.normalize,
    )

    regional_df = extract_regional_features(
        image=image,
        region_mask=region_mask,
        bin_width=args.bin_width,
        normalize=args.normalize,
        include_shape=args.include_shape_in_network,
    )

    build_rrsn(regional_df, args.out_rrsn)

    print(f"Saved: {args.out_radiomics}")
    print(f"Saved: {args.out_rrsn}")


if __name__ == "__main__":
    main()
