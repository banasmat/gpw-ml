import dataset_builder


#dataset_builder.organize_prices_to_quarters()
#dataset = dataset_builder.analyze_dataset()
x, y = dataset_builder.build_dataset(force_reset=True)
#TODO consider filling gaps between values (fillna('ffill')
x, y = dataset_builder.modify_to_diffs(x, y)

print('after',x[:2])
print(x.shape)
print(y)
