from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k2Autoencoder(nn.Module):
    """
    Autoencoder using C3k2 blocks from YOLOv11.
    
    This architecture leverages the efficient C3k2 blocks which provide:
    - CSP (Cross Stage Partial) structure for better gradient flow
    - Faster computation compared to standard convolutions
    - Better parameter efficiency
    
    Architecture follows the standard 4-layer downsampling pattern.
    """
    
    def __init__(self, input_size=(256, 256), latent_dim=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder with C3k2 blocks
        # Initial convolution to expand channels
        self.enc_conv1 = Conv(1, 32, 3, 2, 1)  # /2
        self.enc_c3k2_1 = C3k2(32, 32, n=2, shortcut=True, e=0.5)
        
        self.enc_conv2 = Conv(32, 64, 3, 2, 1)  # /2
        self.enc_c3k2_2 = C3k2(64, 64, n=2, shortcut=True, e=0.5)
        
        self.enc_conv3 = Conv(64, 128, 3, 2, 1)  # /2
        self.enc_c3k2_3 = C3k2(128, 128, n=3, shortcut=True, e=0.5)
        
        self.enc_conv4 = Conv(128, 256, 3, 2, 1)  # /2
        self.enc_c3k2_4 = C3k2(256, 256, n=3, shortcut=True, e=0.5)
        
        # Calculate encoder output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            encoder_output = self._encode(dummy_input)
            self.encoder_output_size = encoder_output.shape[2:]
        
        # Standard bottleneck WITHOUT residual connection (following standard_compact design)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
        )
        
        # Expansion layer to return to 256 channels for decoder
        self.expansion = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        # Decoder with C3k2 blocks
        self.dec_upconv1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)  # x2
        self.dec_c3k2_1 = C3k2(128, 128, n=3, shortcut=True, e=0.5)
        
        self.dec_upconv2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)  # x2
        self.dec_c3k2_2 = C3k2(64, 64, n=2, shortcut=True, e=0.5)
        
        self.dec_upconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)  # x2
        self.dec_c3k2_3 = C3k2(32, 32, n=2, shortcut=True, e=0.5)
        
        self.dec_upconv4 = nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1)  # x2
        
        # Final convolution to reconstruct single channel
        self.final = nn.Conv2d(32, 1, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for conv layers"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _encode(self, x):
        """Encoding path"""
        x = self.enc_conv1(x)
        x = self.enc_c3k2_1(x)
        
        x = self.enc_conv2(x)
        x = self.enc_c3k2_2(x)
        
        x = self.enc_conv3(x)
        x = self.enc_c3k2_3(x)
        
        x = self.enc_conv4(x)
        x = self.enc_c3k2_4(x)
        
        return x
    
    def _decode(self, x):
        """Decoding path"""
        x = self.dec_upconv1(x)
        x = self.dec_c3k2_1(x)
        
        x = self.dec_upconv2(x)
        x = self.dec_c3k2_2(x)
        
        x = self.dec_upconv3(x)
        x = self.dec_c3k2_3(x)
        
        x = self.dec_upconv4(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        # Encode
        encoded = self._encode(x)
        
        # Bottleneck (NO residual connection)
        bottleneck_output = self.bottleneck(encoded)
        
        # Expand back to 256 channels
        expanded = self.expansion(bottleneck_output)
        
        # Decode
        decoded = self._decode(expanded)
        
        # Ensure output matches input size
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final reconstruction
        output = torch.sigmoid(self.final(decoded))
        
        return output
    
    def get_features(self, x):
        """Extract features from the encoder (useful for analysis)"""
        return self._encode(x)
    
    def get_latent(self, x):
        """Extract latent representation (after bottleneck)"""
        encoded = self._encode(x)
        return self.bottleneck(encoded)