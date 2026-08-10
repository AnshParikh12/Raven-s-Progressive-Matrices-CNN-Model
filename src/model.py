import torch
import torch.nn as nn

class CNNEncoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
                        
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(
            256,
            embedding_dim
        )

    def forward(self, x):
        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x

class RavenReasoner(nn.Module):

    def __init__(self, embedding_dim=256, num_heads=8, num_layers=3):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.randn(
                9,
                embedding_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.scorer = nn.Sequential(
            nn.Linear(
                embedding_dim,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1
            )
        )


    def forward(self, context_features, choice_features):
        batch_size = context_features.shape[0]

        scores = []

        for candidate_index in range(8):
            candidate = choice_features[
                :,
                candidate_index,
                :
            ]

            candidate = candidate.unsqueeze(1)

            matrix = torch.cat(
                [
                    context_features,
                    candidate
                ],
                dim=1
            )

            # Add positional information.
            matrix = matrix + self.position_embedding.unsqueeze(0)

            # Relational reasoning
            transformed = self.transformer(
                matrix
            )

            candidate_representation = transformed[
                :,
                8,
                :
            ]

            score = self.scorer(
                candidate_representation
            )

            scores.append(score)

        scores = torch.cat(
            scores,
            dim=1
        )

        return scores