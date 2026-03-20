__all__ = ['RavensDataset']


def __getattr__(name):
    if name == 'RavensDataset':
        from iail_sim_picking.dataset.episode_dataset import RavensDataset

        return RavensDataset
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
