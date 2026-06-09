from monai.losses import DiceFocalLoss

seg_loss = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    reduction="mean",
    batch=True,  # agrège le terme Dice sur le batch : stabilise les structures petites/éparses
)
