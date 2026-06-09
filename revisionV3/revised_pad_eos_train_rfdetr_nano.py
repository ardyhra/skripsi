import torch
from rfdetr import RFDETRNano


def train():
    print("Initializing RF-DETR Nano...")
    model = RFDETRNano(classes=["license_plate"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Starting Training...")
    model.train(
        dataset_dir="data_prepared/lprdataset_rfdetr_coco",
        epochs=20,
        batch_size=4,
        grad_accum_steps=4,
        learning_rate=3e-4,
        output_dir="rfdetr/runs/rfdetr_nano_result_v2",
        device=device,
    )


if __name__ == "__main__":
    train()
