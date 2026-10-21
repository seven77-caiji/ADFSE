import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from Model.AE import AE
from Model.FCMPosition import PositionAE, fit_fcm
from Model.Loss import reconstruction_loss
from Model.Score import compute_errors, compute_scores
from utils import get_device, load_data


class Trainer:
    def __init__(self, config):
        self.config = dict(config)
        self.device = get_device(config['device'])
        # 调用方（SweepRunner）可注入已加载好的数据，避免「每个 k 都重新 load_data + 重新 fit 归一化」。
        # FCM and normalization must use exactly the same normal training subset.
        preloaded = config.get('preloaded')
        if preloaded is not None:
            train, test, self.y_test, self.norm_state = preloaded
        else:
            train, test, self.y_test, self.norm_state = load_data(config)
        self.input_dim = train.shape[1]
        k = config['k']
        if k and (config.get('load_only') or config.get('position_mode') == 'uniform'):
            self.fcm = {'centers': np.zeros((k, self.input_dim), dtype=np.float32)}
        elif k:
            self.fcm = config.get('fcm_snapshot')
            if self.fcm is None:
                self.fcm = fit_fcm(train, k, config['seed'], config['fuzzifier'],
                                        config['fcm_max_iter'], config['fcm_tol'])
        else:
            self.fcm = None
        torch.set_num_threads(1)
        torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        if k:
            self.model = PositionAE(self.input_dim, config['hidden_dims'], config['latent_dim'],
                                    self.fcm['centers'], config['fuzzifier'])
            self.network = self.model.ae
        else:
            self.model = AE(self.input_dim, config['hidden_dims'], config['latent_dim'])
            self.network = self.model
        self.model.to(self.device)
        self.train_x = torch.from_numpy(train).to(self.device)
        self.test_x = torch.from_numpy(test).to(self.device)
        self.train_target = self._target(self.train_x, split='train')
        self.test_target = self._target(self.test_x, split='test')

    @torch.no_grad()
    def _target(self, x, split='train'):
        if not self.config['k']:
            return x
        mode = self.config.get('position_mode', 'fcm')
        if mode == 'uniform':
            membership = torch.full((len(x), self.config['k']), 1.0 / self.config['k'],
                                    device=x.device, dtype=x.dtype)
        elif mode == 'random_train' and split == 'test':
            # Draw only from normal training memberships. Test samples neither
            # contribute to the pool nor influence one another's score.
            rng = np.random.default_rng(self.config['seed'] + 1000004)
            order = rng.integers(0, len(self._train_memberships), size=len(x))
            membership = self._train_memberships[torch.as_tensor(order, device=x.device)]
        else:
            membership = torch.cat([self.model.position(x[i:i+2048])
                                    for i in range(0, len(x), 2048)])
            if mode == 'random_train':
                self._train_memberships = membership
                order = np.random.default_rng(self.config['seed'] + 1000003).permutation(len(x))
                membership = membership[torch.as_tensor(order, device=x.device)]
        return torch.cat([x, membership], dim=1)

    def training(self):
        cfg = self.config
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg['lr'],
                                     weight_decay=cfg['weight_decay'], foreach=False,
                                     fused=self.device.type == 'cuda')
        self.network.train()
        for epoch in range(1, cfg['epochs'] + 1):
            loss_sum = torch.zeros((), device=self.device)
            for start in range(0, len(self.train_target), cfg['batch_size']):
                target = self.train_target[start:start+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self.network(target)['recon']
                if cfg['k']:
                    feature_loss = reconstruction_loss(target[:, :self.input_dim],
                                                       reconstruction[:, :self.input_dim])
                    position_loss = reconstruction_loss(target[:, self.input_dim:],
                                                        reconstruction[:, self.input_dim:])
                    loss = feature_loss + cfg.get('position_loss_weight', 1.0) * position_loss
                else:
                    loss = reconstruction_loss(target, reconstruction)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach() * len(target)
            epoch_loss = (loss_sum / len(self.train_target)).item()
            if not np.isfinite(epoch_loss):
                raise FloatingPointError('Non-finite training loss.')
            if cfg.get('verbose', True) and (epoch == 1 or epoch % 10 == 0):
                print(f"{cfg['dataset_name']} seed={cfg['seed']} k={cfg['k']} epoch={epoch}: loss={epoch_loss:.6f}")

    @torch.no_grad()
    def get_scores(self):
        self.network.eval()
        batches = []
        for start in range(0, len(self.test_target), 4096):
            target = self.test_target[start:start+4096]
            reconstruction = self.network(target)['recon']
            batches.append(compute_scores(target[:, :self.input_dim], reconstruction).cpu().numpy())
        return np.concatenate(batches)

    @torch.no_grad()
    def get_score_parts(self):
        """Return feature and position reconstruction errors for score diagnostics."""
        if not self.config['k']:
            raise ValueError('Position score requires a position model.')
        self.network.eval()
        feature, position = [], []
        for start in range(0, len(self.test_target), 4096):
            target = self.test_target[start:start+4096]
            reconstruction = self.network(target)['recon']
            feature.append((target[:, :self.input_dim] -
                            reconstruction[:, :self.input_dim]).square().mean(dim=1).cpu().numpy())
            position.append((target[:, self.input_dim:] -
                             reconstruction[:, self.input_dim:]).square().mean(dim=1).cpu().numpy())
        return np.concatenate(feature), np.concatenate(position)

    @torch.no_grad()
    def get_errors(self):
        """逐样本逐原始特征的平方误差；其行均值与 `get_scores()` 完全一致。"""
        self.network.eval()
        batches = []
        for start in range(0, len(self.test_target), 4096):
            target = self.test_target[start:start+4096]
            reconstruction = self.network(target)['recon']
            batches.append(compute_errors(target[:, :self.input_dim], reconstruction).cpu().numpy())
        return np.concatenate(batches)

    def evaluate(self):
        scores = self.get_scores()
        return roc_auc_score(self.y_test, scores), average_precision_score(self.y_test, scores)

    def evaluate_errors(self, errors):
        """用已算出的逐特征误差复算指标，避免重复前向。"""
        scores = errors.mean(axis=1)
        return roc_auc_score(self.y_test, scores), average_precision_score(self.y_test, scores)

    def save_run_npz(self, path, errors=None, scores=None, extra=None):
        """保存一次运行的完整产物：测试标签、异常分数（可选逐特征误差）与调用方附加数组。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'y_test': np.asarray(self.y_test, dtype=np.int8)}
        if scores is not None:
            arrays['scores'] = np.asarray(scores, dtype=np.float32)
        if errors is not None:
            arrays['errors'] = np.asarray(errors, dtype=np.float32)
        if extra:
            arrays.update(extra)
        np.savez_compressed(path, **arrays)

    def save_model(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'input_dim': self.input_dim,
                    'hidden_dims': list(self.config['hidden_dims']), 'latent_dim': self.config['latent_dim'],
                    'seed': self.config['seed'], 'k': self.config['k'],
                    'method': 'ADFSE' if self.config['k'] else 'AE',
                    'fuzzifier': self.config['fuzzifier'],
                    'position_loss_weight': self.config.get('position_loss_weight', 1.0),
                    'position_mode': self.config.get('position_mode', 'fcm'),
                    'normalization': {key: torch.from_numpy(value) for key, value in self.norm_state.items()}}, path)

    def load_model(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        for key in ('seed', 'k', 'hidden_dims', 'latent_dim', 'fuzzifier'):
            if state[key] != self.config[key]:
                raise ValueError(f'Checkpoint {key} differs from config: {path}')
        if state.get('position_loss_weight', 1.0) != self.config.get('position_loss_weight', 1.0):
            raise ValueError(f'Checkpoint position_loss_weight differs from config: {path}')
        if state.get('position_mode', 'fcm') != self.config.get('position_mode', 'fcm'):
            raise ValueError(f'Checkpoint position_mode differs from config: {path}')
        if state['method'] != ('ADFSE' if self.config['k'] else 'AE'):
            raise ValueError(f'Checkpoint method differs from config: {path}')
        if state['input_dim'] != self.input_dim:
            raise ValueError(f'Checkpoint input dimension differs: {path}')
        for key, value in self.norm_state.items():
            np.testing.assert_allclose(state['normalization'][key].cpu().numpy(), value)
        self.model.load_state_dict(state['model_state'])
        # Membership targets must use the centers stored in the checkpoint.
        self.train_target = self._target(self.train_x, split='train')
        self.test_target = self._target(self.test_x, split='test')
