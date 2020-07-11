module PlotMoran

export plotPop, plotEntropy, formatFig!


function plotPop(freqs)
    # make sequence frequencies time series.
    fig, ax = subplots(figsize=(6,3))
    formatFig!(fig)
    ax.plot(freqs, lw=0.5)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlim([0,size(freqs,1)])
    ax.set_ylim([0,maximum(freqs)*1.1])
    return fig
end

function plotEntropy(epochTimes::Vector, means::Vector, errors::Vector)
    # make entropy plot vs epoch time plot.
    fig, ax = subplots(figsize=(6,4))
    formatFig!(fig)
    ax.errorbar(epochTimes, means, yerr=errors, fmt="-o", color="k")
    ax.set_xlabel("Epoch Duration", fontsize=14)
    ax.set_ylabel("Sequence Entropy", fontsize=14)
    ax.set_xscale("log")
    return fig
end

function formatFig!(fig)
    # this function sets my plotting preferences
    ax = fig.axes
    for i in 1:length(ax)
        ax[i].set_title("",fontsize=16)
        ax[i].set_xlabel("",fontsize=14)
        ax[i].set_ylabel("",fontsize=14)
        for axs in ["top","bottom","right","left"]
            ax[i].spines[axs].set_linewidth(1.3)
        end
        ax[i].xaxis.set_tick_params(width=1.3)
        ax[i].yaxis.set_tick_params(width=1.3)
    end
    return nothing
end


end
