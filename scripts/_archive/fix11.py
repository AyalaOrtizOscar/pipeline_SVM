df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv", low_memory=False)

# si hay columna is_augment (True/False) o 'orig' marca la original, usa esa. Si no, prioriza label_map_method != NaN
df['is_augment'] = df.get('is_augment', df['basename'].str.contains('_aug|_auto', case=False)).astype(bool)

# ordenar de forma que originales y mapeados queden primero
df = df.sort_values(by=['basename', 'is_augment', df['label_fixed'].notna()], ascending=[True, True, False])

# keep first per basename
df_nd = df.groupby('basename', group_keys=False).first().reset_index()
df_nd.to_csv("D:/pipeline_SVM/features/features_svm_harmonized_dedup_by_basename.csv", index=False)

