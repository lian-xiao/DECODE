import numpy as np
import scipy.spatial

import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import pairwise_distances
import os
from sklearn.preprocessing import QuantileTransformer
from scipy.stats.mstats import winsorize
# from utils.normalize_funcs import standardize_per_catX
# from normalize_funcs import standardize_per_catX

#'dataset_name',['folder_name',[cp_pert_col_name,l1k_pert_col_name],[cp_control_val,l1k_control_val]]
ds_info_dict = {
    "CDRP": ["CDRP-BBBC047-Bray", ["Metadata_Sample_Dose", "pert_sample_dose"]],
    "CDRP-bio": ["CDRPBIO-BBBC036-Bray", ["Metadata_Sample_Dose", "pert_sample_dose"]],
    "TAORF": [
        "TA-ORF-BBBC037-Rohban",
        [
            "Metadata_broad_sample",
            "pert_id",
        ],
    ],
    "LUAD": ["LUAD-BBBC041-Caicedo", ["x_mutation_status", "allele"]],
    "LINCS": ["LINCS-Pilot1", ["Metadata_pert_id_dose", "pert_id_dose"]],
}

labelCol = "PERT"

column_rename_mappings = {
    "CDRP-BBBC047-Bray": {
        "l1k": {
            "pert_id": "Metadata_pert_id",
            "pert_dose": "Metadata_pert_dose_micromolar",
            # "det_plate": "Metadata_Plate",
            "CPD_NAME": "Metadata_pert_iname",
            "CPD_TYPE": "Metadata_cdrp_group",
            "CPD_SMILES": "Metadata_SMILES",
        },
        "cp": {
            # "Metadata_broad_sample": "Metadata_pert_id",
            "Metadata_broad_sample_type": "Metadata_pert_type",
            # "Metadata_mmoles_per_liter2": "Metadata_pert_dose_micromolar",
        },
    },
    "LINCS-Pilot1": {
        "l1k": {
            "pert_dose": "Metadata_pert_dose_micromolar",
            # "det_plate": "Metadata_Plate",
            "cell_id": "Metadata_cell_id",
            "det_well": "Metadata_Well",
            "mfc_plate_name": "Metadata_ARP_ID",
            "pert_iname_x": "Metadata_pert_iname",
            "pert_time": "Metadata_pert_timepoint",
            "pert_mfc_id": "Metadata_pert_id",
            "pert_type_x": "Metadata_pert_type",
            "x_smiles": "Metadata_SMILES",
        },
        "cp": {
            # "Metadata_broad_sample": "Metadata_pert_id",
            # "Metadata_broad_sample_type": "Metadata_pert_type",
            # "Metadata_mmoles_per_liter": "Metadata_pert_dose_micromolar",
            # "pert_iname": "Metadata_pert_iname",
        },
    },}
################################################################################


def adaptive_column_processing(df, features):
    """
    为每列自适应选择最佳的预处理方法
    """
    processed_features = {}
    processing_log = {}
    
    for col in features:
        data = df[col]
        
        # 计算一些统计量来判断数据特性
        q99 = data.quantile(0.99)
        q01 = data.quantile(0.01)
        median = data.median()
        mad = (data - median).abs().median()  # Median Absolute Deviation
        
        # 判断是否存在极端离群值
        # 如果99%分位数与中位数的比值过大，说明存在严重的长尾
        tail_ratio = abs(q99 - median) / (mad + 1e-8)
        
        if tail_ratio > 20:  # 存在严重长尾，需要特殊处理
            # 使用更严格的Winsorization
            processed_data = winsorize(data, limits=[0.05, 0.05])  # 移除上下5%
            processing_log[col] = "Strong Winsorization (0.5%)"
        elif tail_ratio > 10:  # 中等长尾
            processed_data = winsorize(data, limits=[0.01, 0.01])  # 移除上下1%
            processing_log[col] = "Moderate Winsorization (1%)"
        else:  # 分布相对正常
            processed_data = data  # 不做特殊处理
            processing_log[col] = "No Winsorization"
        
        processed_features[col] = processed_data
    
    # 构建处理后的DataFrame
    processed_df = pd.DataFrame(processed_features)
    
    # 标准化
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(processed_df)
    # scaled_df = pd.DataFrame(scaled_features, columns=features)
    # scaler = QuantileTransformer()
    # scaled_features = scaler.fit_transform(processed_df)
    # scaled_df = pd.DataFrame(scaled_features, columns=features)
    
    return processed_df


def vae_data_health_check(data, max_std=10):
    """
    检查数据是否适合VAE训练
    """
    print("=== VAE数据健康检查 ===")
    
    # 检查基本统计量
    print(f"数据形状: {data.shape}")
    print(f"均值范围: [{data.mean().min():.3f}, {data.mean().max():.3f}]")
    print(f"标准差范围: [{data.std().min():.3f}, {data.std().max():.3f}]")
    
    # 检查极值
    abs_max = data.abs().max().max()
    print(f"绝对值最大值: {abs_max:.3f}")
    
    # 检查是否有NaN或inf
    nan_count = data.isna().sum().sum()
    inf_count = np.isinf(data).sum().sum()
    print(f"NaN数量: {nan_count}")
    print(f"Inf数量: {inf_count}")
    
    # 给出建议
    if abs_max > max_std:
        print(f"⚠️  警告: 存在超过±{max_std}的值，可能导致VAE训练不稳定")
        print("建议: 进一步限制离群值或使用梯度裁剪")
    else:
        print("✅ 数据看起来适合VAE训练")
    
    return nan_count == 0 and inf_count == 0 and abs_max <= max_std


def read_replicate_level_profiles(
    dataset_rootDir, dataset, profileType, per_plate_normalized_flag,negcon_normalized_flag=False
):
    """
    Reads replicate level CSV files in the form of a dataframe
    Extract measurments column names for each modalities
    Remove columns with low variance (<thrsh_var)
    Remove columns with more NaNs than a certain threshold (>null_vals_ratio)

    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: if True it will standardize data per plate

    Output:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    """

    dataDir = os.path.join(dataset_rootDir,"preprocessed_data",ds_info_dict[dataset][0])

    cp_data_repLevel = pd.read_csv(
        os.path.join(dataDir,'CellPainting',"replicate_level_cp_" + profileType + ".csv.gz")
    )

    l1k_data_repLevel = pd.read_csv(os.path.join(dataDir, 'L1000',"replicate_level_l1k.csv.gz"))

    cp_features, l1k_features = extract_feature_names(
        cp_data_repLevel, l1k_data_repLevel
    )
    if dataset == 'CDRP':
        drug_df = pd.read_csv('preprocessed_data/CDRP-BBBC047-Bray/CellPainting/chemical_annotations.csv')
        cp_data_repLevel = pd.merge(cp_data_repLevel, drug_df, 
                     left_on='Metadata_broad_sample', 
                     right_on='BROAD_ID', 
                     how='left')  # 使用 inner join,只保留匹配的行

    ########## removes nan and inf values
    l1k_data_repLevel = l1k_data_repLevel.replace([np.inf, -np.inf], np.nan)
    cp_data_repLevel = cp_data_repLevel.replace([np.inf, -np.inf], np.nan)

    #
    null_vals_ratio = 0.05
    thrsh_std = 0.0001
    cols2remove_manyNulls = [
        i
        for i in cp_features
        if (cp_data_repLevel[i].isnull().sum(axis=0) / cp_data_repLevel.shape[0])
        > null_vals_ratio
    ]
    cols2remove_lowVars = (
        cp_data_repLevel[cp_features]
        .std()[cp_data_repLevel[cp_features].std() < thrsh_std]
        .index.tolist()
    )

    cols2removeCP = cols2remove_manyNulls + cols2remove_lowVars
    #     print(cols2removeCP)

    cp_features = list(set(cp_features) - set(cols2removeCP))
    cp_data_repLevel = cp_data_repLevel.drop(cols2removeCP, axis=1)
    cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()

    #     cols2removeCP=[i for i in cp_features if cp_data_repLevel[i].isnull().sum(axis=0)>0]
    #     print(cols2removeCP)

    #     cp=cp.fillna(cp.median())

    # cols2removeGE=[i for i in l1k.columns if l1k[i].isnull().sum(axis=0)>0]
    # print(cols2removeGE)
    # l1k_features = list(set(l1k_features) - set(cols2removeGE))
    # print(len(l1k_features))
    # l1k=l1k.drop(cols2removeGE, axis=1);
    l1k_data_repLevel[l1k_features] = l1k_data_repLevel[l1k_features].interpolate()
    # l1k=l1k.fillna(l1k.median())


    # l1k_data_repLevel[l1k_features] = adaptive_column_processing(l1k_data_repLevel,l1k_features)
    # cp_data_repLevel[cp_features] = adaptive_column_processing(cp_data_repLevel,cp_features)
    
    vae_data_health_check(l1k_data_repLevel[l1k_features])
    vae_data_health_check(cp_data_repLevel[cp_features])


    ############ rename columns that should match to PERT
    labelCol = "PERT"
    cp_data_repLevel = cp_data_repLevel.rename(
        columns={ds_info_dict[dataset][1][0]: labelCol}
    )
    l1k_data_repLevel = l1k_data_repLevel.rename(
        columns={ds_info_dict[dataset][1][1]: labelCol}
    )
    report_pert_overlap(cp_data_repLevel, l1k_data_repLevel, labelCol)

    # qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    # cp_data_repLevel[cp_features] = qt.fit_transform(cp_data_repLevel[cp_features])
    # l1k_data_repLevel[l1k_features] = qt.fit_transform(l1k_data_repLevel[l1k_features])


    
    ################ Per plate scaling
    if per_plate_normalized_flag:
        cp_data_repLevel = standardize_per_catX(
            cp_data_repLevel, "Metadata_Plate", cp_features
        )
        l1k_data_repLevel = standardize_per_catX(
            l1k_data_repLevel, "det_plate", l1k_features
        )

        cols2removeCP = [
            i
            for i in cp_features
            if (cp_data_repLevel[i].isnull().sum(axis=0) / cp_data_repLevel.shape[0])
            > 0.05
        ]
        cp_data_repLevel = cp_data_repLevel.drop(cols2removeCP, axis=1)
        cp_features = list(set(cp_features) - set(cols2removeCP))
        cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()

    if negcon_normalized_flag:
        cp_data_repLevel = normalize_by_negcon_zscore_per_plate(
            cp_data_repLevel, cp_features, "Metadata_Plate"
        )
        l1k_data_repLevel = normalize_by_negcon_zscore_per_plate(
            l1k_data_repLevel, l1k_features, "det_plate"
        )


        ########## rename columns according to column_rename_mappings
    dataset_folder = ds_info_dict[dataset][0]
    if dataset_folder in column_rename_mappings:
        # Rename L1000 columns
        if "l1k" in column_rename_mappings[dataset_folder]:
            l1k_rename_dict = column_rename_mappings[dataset_folder]["l1k"]
            l1k_data_repLevel = l1k_data_repLevel.rename(columns=l1k_rename_dict)
        
        # Rename Cell Painting columns
        if "cp" in column_rename_mappings[dataset_folder]:
            cp_rename_dict = column_rename_mappings[dataset_folder]["cp"]
            cp_data_repLevel = cp_data_repLevel.rename(columns=cp_rename_dict)

    return [cp_data_repLevel, cp_features], [l1k_data_repLevel, l1k_features]


################################################################################
def extract_feature_names(cp_data_repLevel, l1k_data_repLevel):
    """
    extract Cell Painting and L1000 measurments names among the column names

    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data

    Outputs: list of feature names for each modality

    """
    # features to analyse
    cp_features = cp_data_repLevel.columns[
        cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")
    ].tolist()
    l1k_features = l1k_data_repLevel.columns[
        l1k_data_repLevel.columns.str.contains("_at")
    ].tolist()

    return cp_features, l1k_features


################################################################################
def extract_metadata_column_names(cp_data, l1k_data):
    """
    extract metadata column names among the column names for any level of data

    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data

    Outputs: list of metadata column names for each modality

    """
    cp_meta_col_names = cp_data.columns[
        ~cp_data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")
    ].tolist()
    l1k_meta_col_names = l1k_data.columns[
        ~l1k_data.columns.str.contains("_at")
    ].tolist()

    return cp_meta_col_names, l1k_meta_col_names


################################################################################
def read_treatment_level_profiles(
    dataset_rootDir,
    dataset,
    profileType,
    filter_repCorr_params,
    per_plate_normalized_flag,
    negcon_normalized_flag=False,
):
    """
    Reads replicate level CSV files (scaled replicate level profiles per plate)
    Rename the column names to match across datasets to PERT in both modalities
    Remove perturbations with low rep corr across both (filter_perts='highRepOverlap')
            or one of the modalities (filter_perts='highRepUnion')
    Form treatment level profiles by averaging the replicates
    Select and keep the metadata columns you want to keep for each dataset
    Merge treatment level profiles to its own metadata

    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    negcon_normalized_flag: whether to normalize data using negative controls

    Output:
    [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]
    each is a list of dataframe and feature names for each of modalities
    """

    filter_perts = filter_repCorr_params[0]
    repCorrFilePath = filter_repCorr_params[1]

    [cp_data_repLevel, cp_features], [
        l1k_data_repLevel,
        l1k_features,
    ] = read_replicate_level_profiles(
        dataset_rootDir, dataset, profileType, per_plate_normalized_flag, negcon_normalized_flag
    )

    ############ rename columns that should match to PERT
    labelCol = "PERT"
    cp_data_repLevel = cp_data_repLevel.rename(
        columns={ds_info_dict[dataset][1][0]: labelCol}
    )
    l1k_data_repLevel = l1k_data_repLevel.rename(
        columns={ds_info_dict[dataset][1][1]: labelCol}
    )
    report_pert_overlap(cp_data_repLevel, l1k_data_repLevel, labelCol)

    ###### print some data statistics
    print(
        dataset + ": Replicate Level Shapes (nSamples x nFeatures): cp: ",
        cp_data_repLevel.shape[0],
        ",",
        len(cp_features),
        ",  l1k: ",
        l1k_data_repLevel.shape[0],
        ",",
        len(l1k_features),
    )

    print("l1k n of rep: ", l1k_data_repLevel.groupby([labelCol]).size().median())
    print("cp n of rep: ", cp_data_repLevel.groupby([labelCol]).size().median())

    ###### remove perts with low rep corr
    if filter_perts == "highRepOverlap":
        highRepPerts = highRepFinder(dataset, "intersection", repCorrFilePath) + [
            "negcon"
        ]

        cp_data_repLevel = cp_data_repLevel[
            cp_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()
        l1k_data_repLevel = l1k_data_repLevel[
            l1k_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()

    elif filter_perts == "highRepUnion":
        highRepPerts = highRepFinder(dataset, "union", repCorrFilePath) + ["negcon"]

        cp_data_repLevel = cp_data_repLevel[
            cp_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()
        l1k_data_repLevel = l1k_data_repLevel[
            l1k_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()

    ####### form treatment level profiles
    l1k_data_treatLevel = (
        l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index()
    )
    cp_data_treatLevel = (
        cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index()
    )

    ###### define metadata and merge treatment level profiles
    #     dataset:[[cp_columns],[l1k_columns]]
    #     meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
    #                'CDRP-bio':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
    #               'TAORF':[['Metadata_moa'],['pert_type']],
    #               'LUAD':[['Metadata_broad_sample_type','Metadata_pert_type'],[]],
    #               'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}

    meta_dict = {
        "CDRP": [["Metadata_moa", "Metadata_target"], []],
        "CDRP-bio": [["Metadata_moa", "Metadata_target"], []],
        "TAORF": [[], []],
        "LUAD": [[], []],
        "LINCS": [["Metadata_moa", "Metadata_alternative_moa"], ["moa"]],
    }

    meta_cp = (
        cp_data_repLevel[[labelCol] + meta_dict[dataset][0]+['Metadata_Plate']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    meta_l1k = (
        l1k_data_repLevel[[labelCol] + meta_dict[dataset][1]+['Metadata_SMILES','det_plate','Metadata_pert_dose_micromolar']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    cp_data_treatLevel = pd.merge(
        cp_data_treatLevel, meta_cp, how="inner", on=[labelCol]
    )
    l1k_data_treatLevel = pd.merge(
        l1k_data_treatLevel, meta_l1k, how="inner", on=[labelCol]
    )
    report_pert_overlap(cp_data_treatLevel, l1k_data_treatLevel, labelCol)

    return [cp_data_treatLevel, cp_features], [l1k_data_treatLevel, l1k_features]


################################################################################
def read_paired_treatment_level_profiles(
    dataset_rootDir,
    dataset,
    profileType,
    filter_repCorr_params,
    per_plate_normalized_flag,
    add_molformer_features=True,
    smiles_column="Metadata_SMILES",
    add_negcon_features=True,
    remove_negcon_data=True,
    negcon_feature_prefix="NegCon_",
):
    """
    Reads treatment level profiles
    Merge dataframes by PERT column
    Optionally extract and add MoLFormer features from SMILES strings

    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: True for scaling per plate
    negcon_normalized_flag: whether to normalize data using negative controls
    add_molformer_features: whether to extract and add MoLFormer features from SMILES strings
    smiles_column: column name containing SMILES strings, default is 'Metadata_SMILES'
    add_negcon_features: whether to add negative control features as additional columns
    remove_negcon_data: whether to remove negative control data from final dataset
    negcon_feature_prefix: prefix for negative control feature columns

    Output:
    mergedProfiles_treatLevel: paired treatment level profiles
    cp_features: list of Cell Painting feature names
    l1k_features: list of L1000 feature names
    molformer_features: list of MoLFormer feature names (if added)
    dose_features: list of dose feature names
    negcon_features: list of negative control feature names (if added)
    """

    [cp_data_treatLevel, cp_features], [
        l1k_data_treatLevel,
        l1k_features,
    ] = read_treatment_level_profiles(
        dataset_rootDir,
        dataset,
        profileType,
        filter_repCorr_params,
        per_plate_normalized_flag,
    )

    mergedProfiles_treatLevel = pd.merge(
        cp_data_treatLevel, l1k_data_treatLevel, how="inner", on=[labelCol]
    )
    mergedProfiles_treatLevel = mergedProfiles_treatLevel.drop_duplicates().T.drop_duplicates().T
    cp_only_treat, l1k_only_treat = collect_unmatched_profiles(
        cp_data_treatLevel, l1k_data_treatLevel, labelCol
    )
    aligned_treat_profiles, treat_columns, cp_meta_cols, l1k_meta_cols = build_aligned_profile_union(
        mergedProfiles_treatLevel,
        cp_data_treatLevel,
        l1k_data_treatLevel,
        cp_only_treat,
        l1k_only_treat,
        cp_features,
        l1k_features,
        labelCol,
    )
    mergedProfiles_treatLevel.attrs["aligned_profiles"] = aligned_treat_profiles
    mergedProfiles_treatLevel.attrs["aligned_columns"] = treat_columns
    mergedProfiles_treatLevel.attrs["cp_meta_columns"] = cp_meta_cols
    mergedProfiles_treatLevel.attrs["l1k_meta_columns"] = l1k_meta_cols
    print(
        f"单模态Treatment数据：Cell Painting {cp_only_treat.shape[0]} 行，"
        f"L1000 {l1k_only_treat.shape[0]} 行"
    )
    print(
        f"对齐后Treatment总样本: {aligned_treat_profiles.shape[0]} 行，列数 {len(treat_columns)}"
    )

    # 添加阴性对照特征（如果启用）
    cp_negcon_features = []
    l1k_negcon_features = []
    if add_negcon_features:
        print(f"正在添加阴性对照特征...")
        mergedProfiles_treatLevel, cp_negcon_features, l1k_negcon_features = add_negative_control_features(
            mergedProfiles_treatLevel,
            cp_features,
            l1k_features,
            negcon_feature_prefix=negcon_feature_prefix
        )
    
    # 如果需要，提取并添加MoLFormer特征
    molformer_features = []
    if add_molformer_features:
        try:
            from utils.mol_features import add_molformer_features_to_dataframe
            # 检查SMILES列是否存在
            if smiles_column in mergedProfiles_treatLevel.columns:
                print(f"正在从{smiles_column}列提取MoLFormer特征...")
                # 提取特征并添加到DataFrame
                mergedProfiles_treatLevel, molformer_features = add_molformer_features_to_dataframe(
                    mergedProfiles_treatLevel, 
                    smiles_column=smiles_column, 
                )
                
                print(f"已添加{len(molformer_features)}个MoLFormer特征到数据中")
            else:
                print(f"警告: 未找到SMILES列 '{smiles_column}'，无法提取MoLFormer特征")
                print(f"可用的列: {', '.join(mergedProfiles_treatLevel.columns[:10])}...等")
        except Exception as e:
            print(f"提取MoLFormer特征时出错: {e}")
            print("继续处理，但不添加MoLFormer特征")
    
    # 如果需要去除阴性对照数据，在最后处理
    if remove_negcon_data:
        print(f"正在去除阴性对照数据...")
        initial_shape = mergedProfiles_treatLevel.shape
        mergedProfiles_treatLevel = mergedProfiles_treatLevel[
            mergedProfiles_treatLevel[labelCol] != 'negcon'
        ].reset_index(drop=True)
        final_shape = mergedProfiles_treatLevel.shape
        print(f"去除阴性对照数据: {initial_shape} -> {final_shape}")

    dose_features = ['Metadata_pert_dose_micromolar']
    

    print(
        "Treatment Level Shapes (nSamples x nFeatures+metadata):",
        cp_data_treatLevel.shape,
        l1k_data_treatLevel.shape,
        "Merged Profiles Shape:",
        mergedProfiles_treatLevel.shape,
    )

    return mergedProfiles_treatLevel, cp_features, l1k_features, molformer_features, dose_features, cp_negcon_features, l1k_negcon_features


################################################################################
def generate_random_match_of_replicate_pairs(cp_data_repLevel, l1k_data_repLevel, nRep):
    """
    Note that there is no match at the replicate level for this dataset, we either:
        - Forming ALL the possible pairs for replicate level data matching (nRep='all' - string)
        - Randomly sample samples in each modality and form pairs (nRep -> int)

    Inputs:
        cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data

    Outputs:
        Randomly paired replicate level profiles

    """
    labelCol = "PERT"

    if nRep == "all":
        cp_data_n_repLevel = cp_data_repLevel.copy()
        l1k_data_n_repLevel = l1k_data_repLevel.copy()
    else:
        #         nR=np.min((cp_data_repLevel.groupby(labelCol).size().min(),l1k_data_repLevel.groupby(labelCol).size().min()))
        #     cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=nR,replace=True)).reset_index(drop=True)
        nR = nRep
        cp_data_n_repLevel = (
            cp_data_repLevel.groupby(labelCol)
            .apply(lambda x: x.sample(n=np.min([nR, x.shape[0]])))
            .reset_index(drop=True)
        )
        l1k_data_n_repLevel = (
            l1k_data_repLevel.groupby(labelCol)
            .apply(lambda x: x.sample(n=np.min([nR, x.shape[0]])))
            .reset_index(drop=True)
        )

    mergedProfiles_repLevel = pd.merge(
        cp_data_n_repLevel, l1k_data_n_repLevel, how="inner", on=[labelCol], suffixes=('_cp', '_l1k')
    )

    return mergedProfiles_repLevel


################################################################################
def highRepFinder(dataset, how, repCorrFilePath):
    """
    This function reads pre calculated and saved Replicate Correlation values file and filters perturbations
    using one of the following filters:
        - intersection: intersection of high quality profiles across both modalities
        - union: union of high quality profiles across both modalities

    * A High Quality profile is defined as a profile having replicate correlation more than 90th percentile of
      its null distribution

    Inputs:
        dataset (str): dataset name
        how (str):  can be intersection or union

    Output: list of high quality perurbations

    """
    repCorDF = pd.read_excel(repCorrFilePath, sheet_name=None)
    cpRepDF = repCorDF["cp-" + dataset.lower()]
    cpHighList = cpRepDF[cpRepDF["RepCor"] > cpRepDF["Rand90Perc"]][
        "Unnamed: 0"
    ].tolist()
    print("CP: from ", cpRepDF.shape[0], " to ", len(cpHighList))
    cpRepDF = repCorDF["l1k-" + dataset.lower()]
    l1kHighList = cpRepDF[cpRepDF["RepCor"] > cpRepDF["Rand90Perc"]][
        "Unnamed: 0"
    ].tolist()
    #     print("l1kHighList",l1kHighList)
    #     print("cpHighList",cpHighList)
    if how == "intersection":
        highRepPerts = list(set(l1kHighList) & set(cpHighList))
        print("l1k: from ", cpRepDF.shape[0], " to ", len(l1kHighList))
        print("CP and l1k high rep overlap: ", len(highRepPerts))

    elif how == "union":
        highRepPerts = list(set(l1kHighList) | set(cpHighList))
        print("l1k: from ", cpRepDF.shape[0], " to ", len(l1kHighList))
        print("CP and l1k high rep union: ", len(highRepPerts))

    return highRepPerts


################################################################################
def add_negative_control_features(
    df, 
    cp_features, 
    l1k_features, 
    cp_plate_col='Metadata_Plate', 
    l1k_plate_col='det_plate',
    label_col='PERT', 
    negcon_label='negcon',
    negcon_feature_prefix="NegCon_"
):
    """
    为每个样本添加其所在plate的阴性对照特征均值
    
    参数:
    df: 包含特征和标签的数据框
    cp_features: Cell Painting特征列名列表
    l1k_features: L1000特征列名列表
    cp_plate_col: Cell Painting的plate列名
    l1k_plate_col: L1000的plate列名
    label_col: 标签列名，默认为'PERT'
    negcon_label: 阴性对照组的标签，默认为'negcon'
    negcon_feature_prefix: 阴性对照特征的前缀
    
    返回:
    添加了阴性对照特征的数据框和特征名列表
    """
    df_with_negcon = df.copy()
    cp_negcon_features = []
    l1k_negcon_features = []
    print(f"正在为每个样本添加对应plate的阴性对照特征均值...")
    
    # 用于收集新列的字典
    new_columns = {}
    
    # 处理Cell Painting特征
    if cp_plate_col in df.columns and cp_features:
        print(f"处理Cell Painting特征，plate列: {cp_plate_col}")
        
        # 计算每个plate内阴性对照的特征均值 - 使用groupby优化
        negcon_data = df[df[label_col] == negcon_label]
        if negcon_data.empty:
            raise Exception('无法找到Cell Painting阴性对照数据')
        
        # 计算每个plate的阴性对照均值
        cp_plate_negcon_means = negcon_data.groupby(cp_plate_col)[cp_features].mean()
        
        # 获取全局阴性对照均值（用于备用）
        global_cp_means = negcon_data[cp_features].mean()
        
        # 为每个特征创建阴性对照特征列
        for feature in cp_features:
            negcon_feature_name = f"{negcon_feature_prefix}CP_{feature}"
            cp_negcon_features.append(negcon_feature_name)
            
            # 使用map进行向量化操作，更高效
            plate_feature_means = cp_plate_negcon_means[feature].to_dict()
            
            # 对于没有阴性对照的plate，使用全局均值
            new_columns[negcon_feature_name] = df[cp_plate_col].map(
                lambda plate: plate_feature_means.get(plate, global_cp_means[feature])
            )
        
        print(f"已准备 {len(cp_negcon_features)} 个Cell Painting阴性对照特征")
    
    # 处理L1000特征
    if l1k_plate_col in df.columns and l1k_features:
        print(f"处理L1000特征，plate列: {l1k_plate_col}")
        
        # 计算每个plate内阴性对照的特征均值 - 使用groupby优化
        negcon_data = df[df[label_col] == negcon_label]
        if negcon_data.empty:
            raise Exception('无法找到L1000阴性对照数据')
        
        # 计算每个plate的阴性对照均值
        l1k_plate_negcon_means = negcon_data.groupby(l1k_plate_col)[l1k_features].mean()
        
        # 获取全局阴性对照均值（用于备用）
        global_l1k_means = negcon_data[l1k_features].mean()
        
        # 为每个特征创建阴性对照特征列
        for feature in l1k_features:
            negcon_feature_name = f"{negcon_feature_prefix}L1K_{feature}"
            l1k_negcon_features.append(negcon_feature_name)
            
            # 使用map进行向量化操作，更高效
            plate_feature_means = l1k_plate_negcon_means[feature].to_dict()
            
            # 对于没有阴性对照的plate，使用全局均值
            new_columns[negcon_feature_name] = df[l1k_plate_col].map(
                lambda plate: plate_feature_means.get(plate, global_l1k_means[feature])
            )
        
        print(f"已准备 {len(l1k_negcon_features)} 个L1000阴性对照特征")
    
    # 一次性批量添加所有新列，避免DataFrame碎片化
    if new_columns:
        negcon_df = pd.DataFrame(new_columns, index=df_with_negcon.index)
        df_with_negcon = pd.concat([df_with_negcon, negcon_df], axis=1)
        print(f"已批量添加 {len(new_columns)} 个阴性对照特征列")
    
    print(f"已为 {len(df)} 个样本添加了 {len(cp_negcon_features)} 个表型阴性对照特征和 {len(l1k_negcon_features)} 个L1000阴性对照特征")

    return df_with_negcon, cp_negcon_features, l1k_negcon_features


################################################################################
def read_paired_replicate_level_profiles(
    dataset_rootDir,
    dataset,
    profileType,
    nRep,
    filter_repCorr_params,
    per_plate_normalized_flag,
    negcon_normalized_flag=True,
    add_molformer_features=True,
    smiles_column="Metadata_SMILES",
    add_negcon_features=False,
    remove_negcon_data=True,
    negcon_feature_prefix="NegCon_",
):
    """
    Reads replicate level CSV files (scaled replicate level profiles per plate)
    Rename the column names to match across datasets to PERT in both modalities
    Remove perturbations with low rep corr across both (filter_perts='highRepOverlap')
            or one of the modalities (filter_perts='highRepUnion')
    Form treatment level profiles by averaging the replicates
    Select and keep the metadata columns you want to keep for each dataset
    Merge dataframes by PERT column
    Optionally extract and add MoLFormer features from SMILES strings

    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    nRep: number of replicates to use, can be an integer or 'all'
    filter_repCorr_params: parameters for filtering low quality samples
    per_plate_normalized_flag: whether to normalize data per plate
    negcon_normalized_flag: whether to normalize data using negative controls
    add_molformer_features: whether to extract and add MoLFormer features from SMILES strings
    smiles_column: column name containing SMILES strings, default is 'Metadata_SMILES'
    add_negcon_features: whether to add negative control features as additional columns
    remove_negcon_data: whether to remove negative control data from final dataset
    negcon_feature_prefix: prefix for negative control feature columns

    Output:
    mergedProfiles_repLevel: paired replicate level profiles
    cp_features: list of Cell Painting feature names
    l1k_features: list of L1000 feature names
    molformer_features: list of MoLFormer feature names (if added)
    dose_features: list of dose feature names
    negcon_features: list of negative control feature names (if added)
    """

    filter_perts = filter_repCorr_params[0]
    repCorrFilePath = filter_repCorr_params[1]

    [cp_data_repLevel, cp_features], [
        l1k_data_repLevel,
        l1k_features,
    ] = read_replicate_level_profiles(
        dataset_rootDir, dataset, profileType, per_plate_normalized_flag,
        negcon_normalized_flag,
    )

    l1k_data_repLevel, l1k_features = rename_affyprobe_to_genename(l1k_data_repLevel, l1k_features, map_source_address='idmap.xlsx')

    scaler_ge = preprocessing.StandardScaler()
    scaler_cp = preprocessing.StandardScaler()
    l1k_data_repLevel[l1k_features] = scaler_ge.fit_transform(l1k_data_repLevel[l1k_features].values)
    cp_data_repLevel[cp_features] = scaler_cp.fit_transform(cp_data_repLevel[cp_features].values.astype('float64'))

    if 1:
        cp_data_repLevel[cp_features] =preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(cp_data_repLevel[cp_features].values)   
        l1k_data_repLevel[l1k_features] =preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(l1k_data_repLevel[l1k_features].values)           

    ###### print some data statistics
    print(
        dataset + ": Replicate Level Shapes (nSamples x nFeatures): cp: ",
        cp_data_repLevel.shape[0],
        ",",
        len(cp_features),
        ",  l1k: ",
        l1k_data_repLevel.shape[0],
        ",",
        len(l1k_features),
    )

    print("l1k n of rep: ", l1k_data_repLevel.groupby([labelCol]).size().median())
    print("cp n of rep: ", cp_data_repLevel.groupby([labelCol]).size().median())

    ###### remove perts with low rep corr
    if filter_perts == "highRepOverlap":
        highRepPerts = highRepFinder(dataset, "intersection", repCorrFilePath) + [
            "negcon"
        ]

        cp_data_repLevel = cp_data_repLevel[
            cp_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()
        l1k_data_repLevel = l1k_data_repLevel[
            l1k_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()

    elif filter_perts == "highRepUnion":
        highRepPerts = highRepFinder(dataset, "union", repCorrFilePath) + ["negcon"]

        cp_data_repLevel = cp_data_repLevel[
            cp_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()
        l1k_data_repLevel = l1k_data_repLevel[
            l1k_data_repLevel["PERT"].isin(highRepPerts)
        ].reset_index()

    mergedProfiles_repLevel = generate_random_match_of_replicate_pairs(
        cp_data_repLevel, l1k_data_repLevel, nRep
    )
    cp_only_rep, l1k_only_rep = collect_unmatched_profiles(
        cp_data_repLevel, l1k_data_repLevel, labelCol
    )
    aligned_rep_profiles, rep_columns, rep_cp_meta_cols, rep_l1k_meta_cols = build_aligned_profile_union(
        mergedProfiles_repLevel,
        cp_data_repLevel,
        l1k_data_repLevel,
        cp_only_rep,
        l1k_only_rep,
        cp_features,
        l1k_features,
        labelCol,
    )
    # mergedProfiles_repLevel.attrs["aligned_profiles"] = aligned_rep_profiles
    # mergedProfiles_repLevel.attrs["aligned_columns"] = rep_columns
    # mergedProfiles_repLevel.attrs["cp_meta_columns"] = rep_cp_meta_cols
    # mergedProfiles_repLevel.attrs["l1k_meta_columns"] = rep_l1k_meta_cols

    mergedProfiles_repLevel = aligned_rep_profiles # .drop_duplicates().T.drop_duplicates().T
    print(
        f"单模态Replicate数据：Cell Painting {cp_only_rep.shape[0]} 行，"
        f"L1000 {l1k_only_rep.shape[0]} 行"
    )
    print(
        f"对齐后Replicate总样本: {aligned_rep_profiles.shape[0]} 行，列数 {len(rep_columns)}"
    )
    # 添加阴性对照特征（如果启用）
    cp_negcon_features = []
    l1k_negcon_features = []
    if add_negcon_features:
        print(f"正在添加阴性对照特征...")
        mergedProfiles_repLevel, cp_negcon_features, l1k_negcon_features = add_negative_control_features(
            mergedProfiles_repLevel,
            cp_features,
            l1k_features,
            negcon_feature_prefix=negcon_feature_prefix
        )
    


    
    # 如果需要，提取并添加MoLFormer特征
    molformer_features = []
    if add_molformer_features:
        try:
            from utils.mol_features import add_molformer_features_to_dataframe
            # 检查SMILES列是否存在
            if smiles_column in mergedProfiles_repLevel.columns:
                print(f"正在从{smiles_column}列提取MoLFormer特征...")
                # 提取特征并添加到DataFrame
                mergedProfiles_repLevel, molformer_features = add_molformer_features_to_dataframe(
                    mergedProfiles_repLevel, 
                    smiles_column=smiles_column, 
                )
                
                print(f"已添加{len(molformer_features)}个MoLFormer特征到数据中")
            else:
                print(f"警告: 未找到SMILES列 '{smiles_column}'，无法提取MoLFormer特征")
                print(f"可用的列: {', '.join(mergedProfiles_repLevel.columns[:10])}...等")
        except Exception as e:
            print(f"提取MoLFormer特征时出错: {e}")
            print("继续处理，但不添加MoLFormer特征")
    
    
    # 如果需要去除阴性对照数据，在最后处理
    if remove_negcon_data:
        print(f"正在去除阴性对照数据...")
        initial_shape = mergedProfiles_repLevel.shape
        mergedProfiles_repLevel = mergedProfiles_repLevel[
            mergedProfiles_repLevel[labelCol] != 'negcon'
        ].reset_index(drop=True)
        final_shape = mergedProfiles_repLevel.shape
        print(f"去除阴性对照数据: {initial_shape} -> {final_shape}")

    dose_features = ['Metadata_pert_dose_micromolar']
    # 'Cells_Neighbors_FirstClosestDistance_Adjacent', 'Cells_Neighbors_AngleBetweenNeighbors_Adjacent', 'Cells_Neighbors_SecondClosestObjectNumber_Adjacent', 'Cells_Neighbors_SecondClosestDistance_Adjacent', 'Cells_Neighbors_FirstClosestObjectNumber_Adjacent'}
    print('Data Shape:',mergedProfiles_repLevel.shape)
    return mergedProfiles_repLevel, cp_features, l1k_features, molformer_features, dose_features, cp_negcon_features, l1k_negcon_features



def rename_affyprobe_to_genename(l1k_data_df, l1k_features, map_source_address='idmap.xlsx'):
    """
    map input dataframe column name from affy prob id to gene names

    """
    meta = pd.read_excel(map_source_address)

    #     meta=pd.read_csv("../affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
    meta_gene_probID = meta.set_index("probe_id")
    d = dict(zip(meta_gene_probID.index, meta_gene_probID["symbol"]))
    l1k_features_gn = [d[l] for l in l1k_features]
    l1k_data_df = l1k_data_df.rename(columns=d)

    return l1k_data_df, l1k_features_gn



def rename_to_genename_list_to_affyprobe(
    l1k_features_gn, our_l1k_prob_list, map_source_address
):
    """
    map a list of gene names to a list of affy prob ids

    """
    map_source_address='../idmap.xlsx'
    meta = pd.read_excel(map_source_address)
    #     meta=pd.read_csv("../affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
    #     meta=meta[meta['probe_id'].isin(our_l1k_prob_list)].reset_index(drop=True)
    meta_gene_probID = meta.set_index("symbol")
    d = dict(zip(meta_gene_probID.index, meta_gene_probID["probe_id"]))
    l1k_features = [d[l] for l in l1k_features_gn]
    #     l1k_data_df = l1k_data_df.rename(columns=d)

    return l1k_features


def standardize_per_catX(df, column_name, cp_features):
    # column_name='Metadata_Plate'
    #     cp_features=df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
    df_scaled_perPlate = df.copy()
    df_scaled_perPlate[cp_features] = (
        df[cp_features + [column_name]]
        .groupby(column_name)
        .transform(lambda x: (x - x.mean()) / x.std())
        .values
    )
    return df_scaled_perPlate


def normalize_by_negcon_zscore(df, features, label_col='PERT', negcon_label='negcon'):
    """
    使用阴性对照组(negcon)的特征平均值和标准差计算z-score进行标准化
    
    参数:
    df: 包含特征和标签的数据框
    features: 需要标准化的特征列名列表
    label_col: 标签列名，默认为'PERT'
    negcon_label: 阴性对照组的标签，默认为'negcon'
    
    返回:
    标准化后的数据框，其中特征列被z-score标准化
    """
    df_normalized = df.copy()
    
    # 提取阴性对照组数据
    negcon_data = df[df[label_col] == negcon_label]
    
    if negcon_data.empty:
        print(f"警告: 未找到标签为'{negcon_label}'的阴性对照组数据")
        return df_normalized
    
    print(f"使用 {negcon_data.shape[0]} 个阴性对照样本进行标准化")
    
    # 计算阴性对照组的均值和标准差
    negcon_mean = negcon_data[features].mean()
    negcon_std = negcon_data[features].std()
    
    # 处理标准差为0的情况
    negcon_std = negcon_std.replace(0, 1)  # 避免除零错误
    
    # 计算z-score: (x - mean_negcon) / std_negcon
    df_normalized[features] = (df[features] - negcon_mean) / negcon_std
    
    return df_normalized



def normalize_by_negcon_zscore_per_plate(df, features, plate_col, label_col='PERT', negcon_label='negcon'):
    """
    按plate分组，使用每个plate内阴性对照组的特征平均值和标准差计算z-score进行标准化
    
    参数:
    df: 包含特征和标签的数据框
    features: 需要标准化的特征列名列表
    plate_col: plate列名（如'Metadata_Plate'或'det_plate'）
    label_col: 标签列名，默认为'PERT'
    negcon_label: 阴性对照组的标签，默认为'negcon'
    
    返回:
    标准化后的数据框
    """
    df_normalized = df.copy()
    
    for plate in df[plate_col].unique():
        plate_mask = df[plate_col] == plate
        plate_data = df[plate_mask]
        
        # 提取该plate内的阴性对照组数据
        negcon_data = plate_data[plate_data[label_col] == negcon_label]
        
        if negcon_data.empty:
            print(f"警告: Plate {plate} 中未找到阴性对照组数据，跳过标准化")
            continue
        
        # 计算该plate内阴性对照组的均值和标准差
        negcon_mean = negcon_data[features].mean()
        negcon_std = negcon_data[features].std()
        
        # 处理标准差为0的情况
        negcon_std = negcon_std.replace(0, 1)
        
        # 对该plate内的所有样本进行z-score标准化
        df_normalized.loc[plate_mask, features] = (plate_data[features] - negcon_mean) / negcon_std
    
    return df_normalized

def report_pert_overlap(cp_df, l1k_df, label_col="PERT"):
    cp_perts = set(cp_df[label_col].dropna().unique())
    l1k_perts = set(l1k_df[label_col].dropna().unique())
    matched = len(cp_perts & l1k_perts)
    cp_only = len(cp_perts - l1k_perts)
    l1k_only = len(l1k_perts - cp_perts)
    print(
        f"PERT匹配统计：匹配 {matched} 个，"
        f"仅Cell Painting {cp_only} 个，仅L1000 {l1k_only} 个"
    )

def collect_unmatched_profiles(cp_df, l1k_df, label_col="PERT"):
    cp_perts = set(cp_df[label_col].dropna().unique())
    l1k_perts = set(l1k_df[label_col].dropna().unique())
    cp_only = cp_df[cp_df[label_col].isin(cp_perts - l1k_perts)].copy()
    l1k_only = l1k_df[l1k_df[label_col].isin(l1k_perts - cp_perts)].copy()
    if 'CPD_SMILES' in cp_only.columns:
        cp_only['Metadata_SMILES'] = cp_only['CPD_SMILES']
        cp_only.drop(columns=['CPD_SMILES'], inplace=True)
    if 'Metadata_mmoles_per_liter' in cp_only.columns and 'Metadata_pert_dose_micromolar' not in cp_only.columns:
        cp_only['Metadata_pert_dose_micromolar'] = cp_only['Metadata_mmoles_per_liter']
        cp_only.drop(columns=['Metadata_mmoles_per_liter'], inplace=True)
    

    return cp_only, l1k_only

def build_aligned_profile_union(
    matched_df,
    cp_df,
    l1k_df,
    cp_only_df,
    l1k_only_df,
    cp_features,
    l1k_features,
    label_col="PERT",
):
    cp_meta_cols = [c for c in cp_df.columns if c not in cp_features and c != label_col]
    l1k_meta_cols = [c for c in l1k_df.columns if c not in l1k_features and c != label_col]
    full_columns = [label_col]
    for col in matched_df.columns:
        if col not in full_columns:
            full_columns.append(col)

    def _align(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=full_columns)
        df = df.T.drop_duplicates().T
        return df.reindex(columns=full_columns)

    aligned_parts = [
        _align(matched_df),
        _align(cp_only_df),
        _align(l1k_only_df),
    ]
    aligned_df = pd.concat(aligned_parts, ignore_index=True, sort=False)
    return aligned_df, full_columns, cp_meta_cols, l1k_meta_cols
