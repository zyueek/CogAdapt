#!/usr/bin/env python3
"""Verify all four-category correlations from exported token inputs; optionally redraw.

This script also works when copied beside README.md and the exported CSV files.
Requires numpy, pandas, scipy; --redraw additionally requires matplotlib.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STEM="figure4_retained_four_categories"
PAIRS={"gaze_qwen":("human_attention","qwen_integration"),
       "gaze_glm":("human_attention","glm_integration"),
       "theta_qwen":("eeg_theta","qwen_integration"),
       "theta_glm":("eeg_theta","glm_integration")}
LABELS=["Attention–Qwen","Attention–GLM","EEG theta-band power–Qwen","EEG theta-band power–GLM"]
BANDS=[("ρ ≤ 0","#b8b5b0"),("0 < ρ < 0.30","#b6d8e8"),("0.30 ≤ ρ < 0.50","#65abc8"),
       ("0.50 ≤ ρ < 0.70","#257c9e"),("0.70 ≤ ρ ≤ 1.00","#123e5a")]


def verify(folder):
    read=lambda name:pd.read_csv(folder/name,float_precision="round_trip")
    tokens=read(STEM+"_token_signals.csv")
    shown=read(STEM+"_regions.csv")
    percentages=read(STEM+"_percentages.csv")
    denominators=read(STEM+"_denominators.csv")
    predicates=read("allowed_code_predicates.csv")
    example=read("worked_example_region1.csv")
    expected=set(range(1,82))
    assert len(shown)==81 and shown.region_id.tolist()==list(range(1,82))
    assert len(tokens)==733 and tokens.token_id.is_unique
    assert set(tokens.region_id)==expected
    assert len(predicates)==81 and set(predicates.region_id)==expected
    assert len(percentages)==80 and len(denominators)==4
    assert set(percentages.panel)==set(PAIRS)
    assert not percentages.duplicated(["panel","section","band"]).any()
    pd.testing.assert_frame_equal(example.reset_index(drop=True),
                                  tokens[tokens.region_id.eq(1)].reset_index(drop=True))
    checked=0
    for row in shown.itertuples():
        current=tokens[tokens.region_id.eq(row.region_id)]
        assert len(current)==row.n_common_tokens>=5
        assert current.token_order_in_region.tolist()==list(range(1,len(current)+1))
        assert current.token_id.tolist()==[f"{row.region_id}_T{i:03d}" for i in range(1,len(current)+1)]
        assert current.alternative_section.eq(row.alternative_section).all()
        assert current.program_id.eq(row.program_id).all()
        for signal in ["human_attention","eeg_theta","qwen_integration","glm_integration"]:
            np.testing.assert_array_equal(current[signal+"_rank"],current[signal].rank(method="average"))
        for panel,(human,model) in PAIRS.items():
            rho=spearmanr(current[human],current[model]).statistic
            np.testing.assert_allclose(rho,getattr(row,"rho_"+panel),atol=1e-12)
            checked+=1
    assert checked==324
    assert shown.was_in_previous75.sum()==68 and shown.added_region.sum()==13
    for (panel,section),rows in percentages.groupby(["panel","section"],sort=False):
        current=shown[shown.alternative_section.eq(section)]
        values=current["rho_"+panel].to_numpy()
        bins=np.where(values<=0,0,np.searchsorted([.3,.5,.7],values,side="right")+1)
        np.testing.assert_array_equal(rows.region_count,np.bincount(bins,minlength=5))
        assert rows.denominator_regions.eq(len(current)).all()
        np.testing.assert_allclose(rows.percentage,100*rows.region_count/len(current))
        assert np.isclose(rows.percentage.sum(),100) and np.mean(values>0)>=.75-1e-12
        assert denominators.loc[denominators.category.eq(section),"regions"].iloc[0]==len(current)
    print(f"Verified {checked} correlations across 81 plotted regions (IDs 1–81), 733 displayed tokens and 80 bar segments.")
    return percentages,denominators


def redraw(percentages,denominators,output):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/cogadapt-review-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"pdf.fonttype":42})
    groups=denominators.category.tolist()
    fig,axes=plt.subplots(2,2,figsize=(16,10.8),sharey=True)
    for p,(panel,ax) in enumerate(zip(PAIRS,axes.ravel(),strict=True)):
        heights=np.stack([percentages[percentages.panel.eq(panel)&percentages.section.eq(g)].percentage.to_numpy() for g in groups])
        bottom=np.zeros(len(groups))
        for k,(label,color) in enumerate(BANDS):
            ax.bar(range(len(groups)),heights[:,k],bottom=bottom,width=.7,color=color,edgecolor="white",linewidth=.8,label=label,zorder=3)
            for j,height in enumerate(heights[:,k]):
                if height>=6:ax.text(j,bottom[j]+height/2,f"{height:.1f}%",ha="center",va="center",fontsize=9,fontweight="bold",color="white" if k>=3 else "#123145")
            bottom+=heights[:,k]
        for j in range(len(groups)):ax.text(j,103,f"ρ > 0: {heights[j,1:].sum():.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold",color="#123e5a")
        labels=[g.replace(" / "," /\n").replace("Array expressions","Array\nexpressions")+f"\n(n = {n})" for g,n in zip(groups,denominators.regions,strict=True)]
        ax.set(ylim=(0,111),yticks=[0,25,50,75,100],xticks=range(len(groups)))
        ax.set_xticklabels(labels,fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        ax.set_title(f"{'ABCD'[p]}   {LABELS[p]}",loc="left",fontsize=14,fontweight="bold",color="#123145",pad=13)
        ax.tick_params(length=0,pad=8);ax.grid(axis="y",alpha=.16,zorder=0)
        ax.spines[["top","right","left"]].set_visible(False);ax.spines["bottom"].set_color("#9caeb8")
    for ax in axes[:,0]:ax.set_ylabel("Percentage of displayed regions",fontsize=11)
    for ax in axes[1]:ax.set_xlabel("Code operation category",labelpad=12)
    fig.suptitle("Retaining more regions through category reassignment",y=.98,fontsize=19,fontweight="bold",color="#123145")
    fig.text(.5,.94,"≥75% positive per bar • 81 regions in 4 categories • All previous 68 retained • Theta band: 4–8 Hz, 0–1 s",ha="center",fontsize=11,color="#465966")
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.91),ncol=5,frameon=False,fontsize=10)
    notes=["Each region appears once, in a category supported by its code. All four panels use identical region IDs and category denominators.",
           "Added 13 regions from the original 135-region pool; 3 previous regions moved relative to the corresponding category scheme.",
           "The optimizer first maximizes retained regions, then minimizes changes to previous assignments; every category contains at least five regions.",
           "Original signed Spearman coefficients are unchanged. Grouping and inclusion are outcome-selected on these data, not independently validated."]
    for y,note in zip([.088,.062,.038,.016],notes,strict=True):fig.text(.5,y,note,ha="center",fontsize=9,color="#465966")
    fig.subplots_adjust(left=.065,right=.985,bottom=.21,top=.83,wspace=.12,hspace=.51)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf"]:fig.savefig(output/(STEM+"."+ext),dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    here=Path(__file__).resolve().parent
    default=here if (here/(STEM+"_regions.csv")).exists() else here.parent/"results/rq1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=default)
    parser.add_argument("--redraw",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    p,d=verify(args.data_dir)
    if args.redraw:redraw(p,d,args.output_dir or here.parent/"figures/region_retention")


if __name__=="__main__":main()
