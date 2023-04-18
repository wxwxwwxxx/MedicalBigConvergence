from monai.networks.nets import ViT
import torch
from torch import nn
import torch.nn.functional as F
from modules.patch_embedding import PatchEmbedRecover,PixelShuffleUpsampler2D
from monai.networks.layers import Conv, trunc_normal_
__all__ = ["VQ_AE"]
def l2norm(x):
    return F.normalize(x, p=2, dim=-1)


class NaiveAE(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.vit_enc = ViT(1, img_size, patch_size)
        self.input_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 32)
        )
        self.vit_dec = ViT(32, 8, 1, num_layers=3)

    def forward(self, x):
        code = self.vit_enc(x)[0]
        print(code.size())

        code = code.view(-1, 8, 8, 8, 768)
        code = self.input_mlp(code)
        code = torch.permute(code, [0, 4, 1, 2, 3])
        print(code.size())
        out_img = self.vit_dec(code)[0]
        return out_img


class VQ_AE(nn.Module):
    def __init__(self, img_size, patch_size, feature_dim=768, mlp_ratio=4.0 ,num_heads=12,codebook_dim=32, n_code=8192,output_dim=768,beta=0.99):
        super().__init__()
        self.beta = beta
        self.feature_dim = feature_dim
        self.codebook_dim = codebook_dim
        self.output_dim = output_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh(),
            nn.Linear(self.feature_dim, self.codebook_dim)
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh(),
            nn.Linear(self.feature_dim, self.output_dim)
        )
        self.n_patch_dim = img_size // patch_size
        self.vit_dec = ViT(codebook_dim, self.n_patch_dim, 1, num_layers=3,hidden_size=feature_dim,mlp_dim=int(mlp_ratio*feature_dim),num_heads=num_heads)
        self.vit_enc = ViT(1, img_size, patch_size,hidden_size=feature_dim,mlp_dim=int(mlp_ratio*feature_dim),num_heads=num_heads)
        self.mse = nn.MSELoss()
        vq_weight = l2norm(torch.randn(n_code, codebook_dim))

        self.vq_embed = nn.Parameter(vq_weight)

        self.last_vq_weight = self.vq_embed.data.clone()

        self.input_mlp.apply(self._init_weights)
        self.output_mlp.apply(self._init_weights)

    def get_embed_indice(self, input_feature):
        b, p, c = input_feature.size()
        feature = input_feature.view(-1, self.codebook_dim)
        feature = l2norm(feature)
        d = torch.einsum('bd,nd->bn', feature, self.vq_embed)
        d = torch.argmax(d, -1)
        d = d.view(b, self.n_patch_dim, self.n_patch_dim, self.n_patch_dim)
        return d

    def get_embed_feature(self, indice):
        embed = F.embedding(indice, self.vq_embed)
        embed = torch.einsum("bhwdc->bchwd", embed)
        return embed
    def codebook_update(self):
        self.last_vq_weight = self.last_vq_weight.to(self.vq_embed.data.device)
        self.vq_embed.data = l2norm(self.beta*self.vq_embed.data+(1.0-self.beta)*self.last_vq_weight)
        self.last_vq_weight = self.vq_embed.data.clone()
    def forward(self, img):
        encoder_f = self.vit_enc(img)[0]
        encoder_downsample_f = self.input_mlp(encoder_f)
        indice = self.get_embed_indice(encoder_downsample_f)
        codebook_f = self.get_embed_feature(indice)
        b, p, c = encoder_downsample_f.size()
        # spa is short for spatial
        encoder_downsample_f_spa = encoder_downsample_f.view(b, self.n_patch_dim, self.n_patch_dim, self.n_patch_dim, c)
        encoder_downsample_f_spa = torch.einsum("bhwdc->bchwd", encoder_downsample_f_spa)
        grad_bridge = encoder_downsample_f_spa + (codebook_f - encoder_downsample_f_spa).detach()

        decode_f = self.vit_dec(grad_bridge)[0]
        l_embed2encoder = self.mse(codebook_f, encoder_downsample_f_spa.detach())
        l_encoder2embed = self.mse(encoder_downsample_f_spa, codebook_f.detach())

        decode_upsample_f = self.output_mlp(decode_f)
        return decode_upsample_f, l_encoder2embed, l_embed2encoder, indice

    def forward_warmup(self, img):
        encoder_f = self.vit_enc(img)[0]
        encoder_downsample_f = self.input_mlp(encoder_f)
        b, p, c = encoder_downsample_f.size()
        # spa is short for spatial
        encoder_downsample_f_spa = encoder_downsample_f.view(b, self.n_patch_dim, self.n_patch_dim, self.n_patch_dim, c)
        encoder_downsample_f_spa = torch.einsum("bhwdc->bchwd", encoder_downsample_f_spa)
        #grad_bridge = encoder_downsample_f_spa + (codebook_f - encoder_downsample_f_spa).detach()

        decode_f = self.vit_dec(encoder_downsample_f_spa)[0]
        # l_embed2encoder = self.mse(codebook_f, encoder_downsample_f_spa.detach())
        # l_encoder2embed = self.mse(encoder_downsample_f_spa, codebook_f.detach())

        decode_upsample_f = self.output_mlp(decode_f)
        return decode_upsample_f
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class VQ_AE_2D(nn.Module):
    def __init__(self, img_size, patch_size, feature_dim=768, mlp_ratio=4.0 ,num_heads=12,codebook_dim=32, n_code=8192,output_dim=768,beta=0.99):
        super().__init__()
        self.beta = beta
        self.feature_dim = feature_dim
        self.codebook_dim = codebook_dim
        self.output_dim = output_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh(),
            nn.Linear(self.feature_dim, self.codebook_dim)
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh(),
            nn.Linear(self.feature_dim, self.output_dim)
        )
        self.n_patch_dim = img_size // patch_size
        self.vit_dec = ViT(codebook_dim, self.n_patch_dim, 1, num_layers=3,hidden_size=feature_dim,mlp_dim=int(mlp_ratio*feature_dim),num_heads=num_heads,spatial_dims=2)
        self.vit_enc = ViT(1, img_size, patch_size,hidden_size=feature_dim,mlp_dim=int(mlp_ratio*feature_dim),num_heads=num_heads,spatial_dims=2)
        self.mse = nn.MSELoss()
        vq_weight = l2norm(torch.randn(n_code, codebook_dim))

        self.vq_embed = nn.Parameter(vq_weight)

        self.last_vq_weight = self.vq_embed.data.clone()

        self.input_mlp.apply(self._init_weights)
        self.output_mlp.apply(self._init_weights)
        self.debug_log = True
    def get_embed_indice(self, input_feature):
        b, p, c = input_feature.size()
        feature = input_feature.view(-1, self.codebook_dim)
        # feature = l2norm(feature) #L2 norm is move to "forward" function
        d = torch.einsum('bd,nd->bn', feature, self.vq_embed)
        d = torch.argmax(d, -1)
        d = d.view(b, self.n_patch_dim, self.n_patch_dim)
        return d

    def get_embed_feature(self, indice):
        embed = F.embedding(indice, self.vq_embed)
        embed = torch.einsum("bhwc->bchw", embed)
        return embed
    def codebook_update(self):
        self.last_vq_weight = self.last_vq_weight.to(self.vq_embed.data.device)
        self.vq_embed.data = l2norm(self.beta*self.vq_embed.data+(1.0-self.beta)*self.last_vq_weight)
        self.last_vq_weight = self.vq_embed.data.clone()
    def forward(self, img):
        encoder_f = self.vit_enc(img)[0]
        encoder_downsample_f = self.input_mlp(encoder_f)
        encoder_downsample_f = l2norm(encoder_downsample_f)
        indice = self.get_embed_indice(encoder_downsample_f)
        codebook_f = self.get_embed_feature(indice)
        b, p, c = encoder_downsample_f.size()
        # spa is short for spatial
        encoder_downsample_f_spa = encoder_downsample_f.view(b, self.n_patch_dim, self.n_patch_dim, c)
        encoder_downsample_f_spa = torch.einsum("bhwc->bchw", encoder_downsample_f_spa)
        grad_bridge = encoder_downsample_f_spa + (codebook_f - encoder_downsample_f_spa).detach()

        decode_f = self.vit_dec(grad_bridge)[0]
        l_embed2encoder = self.mse(codebook_f, encoder_downsample_f_spa.detach())
        l_encoder2embed = self.mse(encoder_downsample_f_spa, codebook_f.detach())

        decode_upsample_f = self.output_mlp(decode_f)
        return decode_upsample_f, l_encoder2embed, l_embed2encoder, indice

    def forward_warmup(self, img):
        encoder_f,hs = self.vit_enc(img)
        if self.debug_log:
            print(f"\033[31mDebug Info:Hidden state num:{len(hs)},Hidden state size:{hs[0].size()}\033[0m")
            self.debug_log = False
        encoder_downsample_f = self.input_mlp(encoder_f)
        encoder_downsample_f = l2norm(encoder_downsample_f)
        b, p, c = encoder_downsample_f.size()
        # spa is short for spatial
        encoder_downsample_f_spa = encoder_downsample_f.view(b, self.n_patch_dim, self.n_patch_dim, c)

        encoder_downsample_f_spa = torch.einsum("bhwc->bchw", encoder_downsample_f_spa)
        #grad_bridge = encoder_downsample_f_spa + (codebook_f - encoder_downsample_f_spa).detach()

        decode_f = self.vit_dec(encoder_downsample_f_spa)[0]
        # l_embed2encoder = self.mse(codebook_f, encoder_downsample_f_spa.detach())
        # l_encoder2embed = self.mse(encoder_downsample_f_spa, codebook_f.detach())

        decode_upsample_f = self.output_mlp(decode_f)
        return decode_upsample_f
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

if __name__ == "__main__":
    s = VQ_AE_2D(512, 32)
    a = PixelShuffleUpsampler2D(768)
    # b = ConvTransUpsampler(768)
    # c = TrilinearUpsampler(768)

    r = PatchEmbedRecover(512,32,a,hidden_size=768,cls=False,spatial_dims=2)
    optim = torch.optim.AdamW([*s.parameters()],lr=1e-3)
    loss_recon = torch.nn.MSELoss()

    for i in range(100):
        print(i)
        optim.zero_grad()
        t_input = torch.rand([4, 1, 512,512])
        pred = s.forward_warmup(t_input)
        print(pred.size())
        recon = r(pred)
        print(recon.shape)
        # l = loss_recon(recon, a) + l2 + l3
        # print(l.item(),l2.item(),l3.item())
        # l.backward()
        # optim.step()
        # s.norm_codebook()
        # print(torch.einsum("ab,ab->a",s.vq_embed,s.vq_embed))

    # for n, i in s.named_parameters():
    #     print(n, i.grad.size() if i.grad is not None else None)
