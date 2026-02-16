import torch

from src.backbone.ireshyper import ir_50_hyper


def load_model(model, checkpoint_path, device="cpu"):
    """
    model_class: Python-Klasse deines Modells (z.B. MyModel)
    checkpoint_path: Pfad zur .pt oder .pth Datei
    """
    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def export_to_onnx(model, dummy_input, output_path, opset=17):
    """
    input model: PyTorch Model
    dummy_input: z.B. torch.randn(1, 3, 112, 112)
    output_path: for .onnx File
    opset: ONNX Opset Version
    """
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    #from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18, iresnet100
    #checkpoint = "F:\\Face\\HM_IDENT_3DFR\\pretrained\\AdaFace_ARoFace_R100_WebFace12M.pt"
    #onnx_path = "F:\\Face\\HM_IDENT_3DFR\\pretrained\\AdaFace_ARoFace_R100_WebFace12M.onnx"
    #device = "cpu"
    #model = load_model(iresnet100(embedding_size=512, fp16=False), checkpoint, device)

    #dummy = torch.randn(1, 3, 112, 112, device=device)
    #export_to_onnx(model, dummy, onnx_path)


    checkpoint = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\HyperFace50K_ir50_adaface.ckpt"
    onnx_path = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\HyperFace50K_ir50_adaface.onnx"
    device = "cpu"

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]
    new_state_dict = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
    }
    new_state_dict = {
        k: v
        for k, v in new_state_dict.items()
        if not k.startswith("head.")
    }
    model = ir_50_hyper(embedding_size=512)
    model.load_state_dict(new_state_dict)
    dummy = torch.randn(1, 3, 112, 112, device=device)
    export_to_onnx(model, dummy, onnx_path)
