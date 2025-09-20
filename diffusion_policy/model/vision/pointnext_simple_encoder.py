import torch
import torch.nn as nn
from pointnext import PointNext, pointnext_s, pointnext_b, pointnext_l, pointnext_xl

class PointNeXtSimpleEncoder(nn.Module):

    def __init__(self, model_name, in_dim=6, out_dim=512, dropout=0.1):
        super().__init__()

        if model_name == 'pointnext_s':
            encoder = pointnext_s(in_dim=in_dim)
        elif model_name == 'pointnext_b':
            encoder = pointnext_b(in_dim=in_dim)
        elif model_name == 'pointnext_l':
            encoder = pointnext_l(in_dim=in_dim)
        elif model_name == 'pointnext_xl':
            encoder = pointnext_xl(in_dim=in_dim)
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented for PointNeXtSimpleEncoder.")
        self.backbone = PointNext(512, encoder=encoder)

        self.norm = nn.BatchNorm1d(512)
        self.act = nn.ReLU()
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, xyz):
        out = self.norm(self.backbone(x.permute(0,2,1), xyz.permute(0,2,1)))
        out = out.mean(dim=-1)
        out = self.act(out)
        out = self.cls_head(out)
        return out


if __name__ == "__main__":
    # Example usage
    import pdb; pdb.set_trace()
    model = PointNeXtSimpleEncoder(model_name='pointnext_s').to('cuda')
    x = torch.randn(8, 1024, 6).to('cuda')  # Batch of 8, 6 features, 1024 points
    xyz = torch.randn(8, 1024, 3).to('cuda')  # Corresponding XYZ coordinates
    output = model(x, xyz)
    print(output.shape)  # Should print torch.Size([8, out_dim])

# python -m diffusion_policy.model.vision.pointnext_simple_encoder