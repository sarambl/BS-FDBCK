import xarray as xr
from matplotlib import pyplot as plt


def plot_seasonal_plots(da,
                        varname=None,
                        x=None,
                        y=None,
                        seasons=None,
                        axs=None,
                        label=None,
                        title=None,
                        sharex=True,
                        sharey=True,
                        figsize=None,
                        legend=True,
                        apply_method=None,
                        **kwargs
                        ):
    if isinstance(da, xr.Dataset):
        da = da[varname]
    if apply_method is None:
        apply_method = xr.DataArray.mean
    _da = apply_method(da.groupby('time.season'))  # .mean('time')
    if seasons is None:
        try:
            seasons = list(_da.coords['season'].values)
        except TypeError:
            seasons = [str(_da.coords['season'].values)]
    if axs is None:
        fig, axs = plt.subplots(2, 2, sharex=sharex, sharey=sharey, figsize=figsize)
    for seas, ax in zip(seasons, axs.flatten()):

        if label is not None:
            kwargs['label'] = label + ', ' + seas

        _da.sel(season=seas).plot(x=x, y=y, ax=ax, **kwargs)
        if title is not None:
            tit = title + ', ' + seas
        else:
            tit = seas
        ax.set_title(tit)
    if legend:
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left',
        #   ncol=2, mode="expand", borderaxespad=0.)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.show()
