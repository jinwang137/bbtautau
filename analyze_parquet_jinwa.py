#!/usr/bin/env python3
"""
分析 coffea skimmer 输出的 parquet 文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def load_parquet_files(pattern):
    """加载所有匹配的 parquet 文件"""
    files = glob.glob(pattern)
    print(f"Found {len(files)} parquet files")
    
    dataframes = []
    for file in files:
        print(f"Loading {file}")
        df = pd.read_parquet(file)
        dataframes.append(df)
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total events: {len(combined_df)}")
        return combined_df
    else:
        print("No files found!")
        return None

def print_summary(df):
    """打印数据摘要"""
    print("\n=== Data Summary ===")
    print(f"Total events: {len(df)}")
    
    # 修复weight sum的格式化问题
    if 'weight' in df.columns:
        weight_sum = float(df['weight'].sum())
        print(f"Weighted events: {weight_sum:.2f}")
    
    # 查看前几行数据
    print(f"\n=== First 3 rows ===")
    print(df.head(3))
    
    # 查看变量
    print(f"\n=== Available columns ({len(df.columns)}) ===")
    
    # 按类型分组
    fatjet_vars = [col for col in df.columns if 'ak8FatJet' in col]
    jet_vars = [col for col in df.columns if 'ak4Jet' in col]
    lepton_vars = [col for col in df.columns if any(lep in col for lep in ['Electron', 'Muon', 'Tau'])]
    weight_vars = [col for col in df.columns if 'weight' in col]
    
    # 输出所有变量名和类型
    print(f"\n  FatJet variables ({len(fatjet_vars)}):")
    for var in fatjet_vars:
        print(f"    {var}: {df[var].dtype}")
    
    print(f"\n  AK4Jet variables ({len(jet_vars)}):")
    for var in jet_vars:
        print(f"    {var}: {df[var].dtype}")
    
    print(f"\n  Lepton variables ({len(lepton_vars)}):")
    for var in lepton_vars:
        print(f"    {var}: {df[var].dtype}")
    
    print(f"\n  Weight variables ({len(weight_vars)}):")
    for var in weight_vars:
        print(f"    {var}: {df[var].dtype}")
    
    # 其他变量
    other_vars = [col for col in df.columns if col not in fatjet_vars + jet_vars + lepton_vars + weight_vars]
    print(f"\n  Other variables ({len(other_vars)}):")
    for var in other_vars:
        print(f"    {var}: {df[var].dtype}")
    
    # 显示数据类型
    print(f"\n=== Data types ===")
    print(df.dtypes.value_counts())

def plot_distributions(df, output_dir="plots"):
    """绘制主要变量的分布"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    weights = df['weight'] if 'weight' in df.columns else None
    
    # 简单检查一些关键变量
    plt.figure(figsize=(12, 8))
    
    # 找到可用的变量并绘制
    plot_count = 0
    
    # FatJet mass
    
    # FatJet matching
    if 'ak8FatJetCAmatched_2BoostedTaus' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        if len(df['ak8FatJetCAmatched_2BoostedTaus'].shape) > 1:
            data = df['ak8FatJetCAmatched_2BoostedTaus'].iloc[:, 0]
        else:
            data = df['ak8FatJetCAmatched_2BoostedTaus']
        
        plt.hist(data, bins=50, alpha=0.7, weights=weights)
        plt.xlabel('ak8FatJetCAmatched_2BoostedTaus')
        plt.ylabel('Events')
        plt.title('ak8FatJetCAmatched_2BoostedTaus')
    
    # FatJet matching
    if 'ak8FatJetCAmatched_2BoostedTaus' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        if len(df['ak8FatJetCAmatched_2BoostedTaus'].shape) > 1:
            data = df['ak8FatJetCAmatched_2BoostedTaus'].iloc[:, 1]
        else:
            data = df['ak8FatJetCAmatched_2BoostedTaus']
        
        plt.hist(data, bins=50, alpha=0.7, weights=weights)
        plt.xlabel('ak8FatJetCAmatched_2BoostedTaus (2nd FatJet)')
        plt.ylabel('Events')
        plt.title('ak8FatJetCAmatched_2BoostedTaus (2nd FatJet)')

    # Number of jets
    if 'nFatJets' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        plt.hist(df['nFatJets'], bins=10, alpha=0.7, weights=weights)
        plt.xlabel('Number of FatJets')
        plt.ylabel('Events')
        plt.title('N(FatJets)')
    
    # Weight distribution
    if 'CA_matched_tau_pt_sum' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        plt.hist(df['CA_matched_tau_pt_sum'], bins=10, alpha=0.7, weights=weights)
        plt.xlabel('CA_matched_tau_pt_sum')
        plt.ylabel('Events')
        plt.title('CA_matched_tau_pt_sum')

    if 'CA_tau_idx_0' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        plt.hist(df['CA_tau_idx_0'], bins=10, alpha=0.7, weights=weights)
        plt.xlabel('CA_tau_idx_0')
        plt.ylabel('Events')
        plt.title('CA_tau_idx_0')

    if 'CA_best_fatjet_idx' in df.columns:
        plot_count += 1
        plt.subplot(2, 3, plot_count)
        plt.hist(df['CA_best_fatjet_idx'], bins=10, alpha=0.7, weights=weights)
        plt.xlabel('CA_best_fatjet_idx')
        plt.ylabel('Events')
        plt.title('CA_best_fatjet_idx')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quick_check_add3.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_dir}/quick_check_add3.png")
    plt.show()

def analyze_selections(df):
    """分析选择效率"""
    print("\n=== Selection Analysis ===")
    
    total_events = len(df)
    total_weighted = np.sum(df['weight'])
    
    # 定义一些选择
    selections = {}
    
    if 'ak8FatJetPt' in df.columns:
        selections['FatJet pT > 250'] = df['ak8FatJetPt'].iloc[:, 0] > 250
        selections['FatJet pT > 300'] = df['ak8FatJetPt'].iloc[:, 0] > 300
    
    if 'ak8FatJetMsd' in df.columns:
        selections['FatJet mSD > 50'] = df['ak8FatJetMsd'].iloc[:, 0] > 50
        selections['FatJet mSD > 100'] = df['ak8FatJetMsd'].iloc[:, 0] > 100
    
    if 'ak8FatJetParTXbbvsQCD' in df.columns:
        selections['ParT Xbb > 0.3'] = df['ak8FatJetParTXbbvsQCD'].iloc[:, 0] > 0.3
        selections['ParT Xbb > 0.8'] = df['ak8FatJetParTXbbvsQCD'].iloc[:, 0] > 0.8
    
    if 'METPt' in df.columns:
        selections['MET > 50'] = df['METPt'] > 50
        selections['MET > 100'] = df['METPt'] > 100
    
    # 计算效率
    print(f"{'Selection':<20} {'Events':<10} {'Efficiency':<12} {'Weighted Events':<15} {'Weighted Eff':<15}")
    print("-" * 80)
    print(f"{'Total':<20} {total_events:<10} {'100.0%':<12} {total_weighted:<15.2f} {'100.0%':<15}")
    
    for name, selection in selections.items():
        passed_events = np.sum(selection)
        passed_weighted = np.sum(df['weight'][selection])
        eff = 100 * passed_events / total_events
        eff_weighted = 100 * passed_weighted / total_weighted
        
        print(f"{name:<20} {passed_events:<10} {eff:<12.1f}% {passed_weighted:<15.2f} {eff_weighted:<15.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze parquet files from coffea skimmer')
    parser.add_argument('--input', '-i', default='*.parquet', 
                       help='Pattern for input parquet files (default: *.parquet)')
    parser.add_argument('--output', '-o', default='plots', 
                       help='Output directory for plots (default: plots)')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_parquet_files(args.input)
    if df is None:
        return
    
    # 打印摘要
    print_summary(df)
    
    # 绘制分布 - 简化版本
    plot_distributions(df, args.output)
    
    print(f"\nQuick check complete! Plot saved to {args.output}/")

if __name__ == "__main__":
    main()
