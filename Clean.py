
import pandas as pd
import numpy as np
import difflib

train_df=pd.read_csv('train.csv')
train_df.info()


train_df.isna().sum()


test_df=pd.read_csv('test.csv')
test_df.info()


test_df.isna().sum()

train_df['category'].value_counts()
test_df['category'].value_counts()

a=train_df['category'].value_counts()
b=test_df['category'].value_counts()


c=train_df['sub_category'].value_counts()
d=test_df['sub_category'].value_counts()

train_df['sub_category'].value_counts()
test_df['sub_category'].value_counts()







category_missing = train_df.groupby('category')['sub_category'].apply(lambda x: x.isnull().sum()).sort_values(ascending=False)


subcategory_missing = train_df.groupby('sub_category')['crimeaditionalinfo'].apply(lambda x: x.isnull().sum()).sort_values(ascending=False)

print("Missing sub_category counts by category:")
print(category_missing)

print("\nMissing crimeaditionalinfo counts by sub_category:")
print(subcategory_missing)



import pandas as pd


category_missing = test_df.groupby('category')['sub_category'].apply(lambda x: x.isnull().sum()).sort_values(ascending=False)


subcategory_missing = train_df.groupby('sub_category')['crimeaditionalinfo'].apply(lambda x: x.isnull().sum()).sort_values(ascending=False)

print("Missing sub_category counts by category:")
print(category_missing)

print("\nMissing crimeaditionalinfo counts by sub_category:")
print(subcategory_missing)



import pandas as pd


# Mapping for missing sub_category values based on category
missing_mapping = {
    "RapeGang Rape RGRSexually Abusive Content": "Rape/Gang Rape-Sexually Abusive Content",
    "Sexually Obscene material": "Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material",
    "Sexually Explicit Act": "Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material",
    "Child Pornography CPChild Sexual Abuse Material CSAM": "Child Pornography/Child Sexual Abuse Material (CSAM)"
}

# Fill missing sub_category based on the category
train_df['sub_category'] = train_df.apply(
    lambda row: missing_mapping.get(row['category'], row['sub_category']),
    axis=1
)


train_df = train_df.dropna(subset=['crimeaditionalinfo'])


train_df.to_csv('cleaned_train_data.csv', index=False)

print("Data cleaning complete. The cleaned file is saved as 'cleaned_train_data.csv'.")



df=pd.read_csv('cleaned_train_data.csv')
df.info()


df.isna().sum()


import pandas as pd

y
missing_mapping = {
    "RapeGang Rape RGRSexually Abusive Content": "Rape/Gang Rape-Sexually Abusive Content",
    "Sexually Obscene material": "Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material",
    "Sexually Explicit Act": "Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material",
    "Child Pornography CPChild Sexual Abuse Material CSAM": "Child Pornography/Child Sexual Abuse Material (CSAM)"
}


test_df['sub_category'] = test_df.apply(
    lambda row: missing_mapping.get(row['category'], row['sub_category']),
    axis=1
)


test_df = test_df.dropna(subset=['crimeaditionalinfo'])


test_df.to_csv('cleaned_test_data.csv', index=False)

print("Data cleaning complete. The cleaned file is saved as 'cleaned_train_data.csv'.")



df=pd.read_csv('cleaned_test_data.csv')
df.info()


df.isna().sum()
df['sub_category'].value_counts()




def analyze_category_differences(train_df, test_df):
    
    print("Category Comparison:")
    train_categories = set(train_df['category'].unique())
    test_categories = set(test_df['category'].unique())
    
    print("Categories in Training Set:")
    print(train_categories)
    print("\nCategories in Test Set:")
    print(test_categories)
    
    print("\nCategories in Test but not in Train:")
    print(test_categories - train_categories)
    print("\nCategories in Train but not in Test:")
    print(train_categories - test_categories)
    
   
    print("\n\nSubcategory Comparison:")
    train_subcategories = set(train_df['sub_category'].unique())
    test_subcategories = set(test_df['sub_category'].unique())
    
    print("Subcategories in Test but not in Train:")
    unique_test_subcategories = test_subcategories - train_subcategories
    print(unique_test_subcategories)
    
    print("\nSubcategories in Train but not in Test:")
    unique_train_subcategories = train_subcategories - test_subcategories
    print(unique_train_subcategories)
    
   
    def find_closest_matches(unique_items, full_set, threshold=0.6):
        matches = {}
        for item in unique_items:
            # Find closest match
            closest = difflib.get_close_matches(item, full_set, n=1, cutoff=threshold)
            if closest:
                matches[item] = closest[0]
        return matches
    
 
    print("\nClose Category Matches:")
    category_matches = find_closest_matches(
        list(test_categories - train_categories), 
        list(train_categories)
    )
    print(category_matches)
    
    print("\nClose Subcategory Matches:")
    subcategory_matches = find_closest_matches(
        list(unique_test_subcategories), 
        list(train_subcategories)
    )
    print(subcategory_matches)
    
    return {
        'category_differences': test_categories - train_categories,
        'subcategory_differences': unique_test_subcategories,
        'category_matches': category_matches,
        'subcategory_matches': subcategory_matches
    }


alignment_results = analyze_category_differences(train_df, test_df)

def create_category_mapping_strategy(train_df, test_df, alignment_results):
   
    category_mapping = {}
    
  
    category_mapping.update(alignment_results['category_matches'])
    
   
    manual_category_corrections = {
        'Crime Against Women & Children': 'RapeGang Rape RGRSexually Abusive Content',
        
    }
    category_mapping.update(manual_category_corrections)
    
    
    subcategory_mapping = {}
    
   
    subcategory_mapping.update(alignment_results['subcategory_matches'])
    
   
    manual_subcategory_corrections = {
        'Computer Generated CSAM/CSEM': 'Child Pornography CPChild Sexual Abuse Material CSAM',
        'Cyber Blackmailing & Threatening': 'Other',
        'Sexual Harassment': 'RapeGang Rape RGRSexually Abusive Content',
       
    }
    subcategory_mapping.update(manual_subcategory_corrections)
    
  
    def map_categories_and_subcategories(row, mapping_type):
        if mapping_type == 'category':
            mapping_dict = category_mapping
            column = 'category'
        else:
            mapping_dict = subcategory_mapping
            column = 'sub_category'
        
        
        if row[column] in mapping_dict:
            return mapping_dict[row[column]]
        
     
        return row[column]
    
    
    test_df['category'] = test_df.apply(
        lambda row: map_categories_and_subcategories(row, 'category'), 
        axis=1
    )
    test_df['sub_category'] = test_df.apply(
        lambda row: map_categories_and_subcategories(row, 'subcategory'), 
        axis=1
    )
    
    return test_df


test_df = create_category_mapping_strategy(train_df, test_df, alignment_results)


def validate_mapping(original_test_df, mapped_test_df):
    print("\nMapping Validation:")
    print("Original Test Set Categories:")
    print(original_test_df['category'].value_counts())
    
    print("\nMapped Test Set Categories:")
    print(mapped_test_df['category'].value_counts())
    
    print("\nOriginal Test Set Subcategories:")
    print(original_test_df['sub_category'].value_counts())
    
    print("\nMapped Test Set Subcategories:")
    print(mapped_test_df['sub_category'].value_counts())


validate_mapping(test_df.copy(), test_df)


#############################################################################################

# Group data by category and aggregate unique subcategories
category_subcategory_mapping = train_df.groupby('category')['sub_category'].unique()


for category, subcategories in category_subcategory_mapping.items():
    print(f"Category: {category}")
    print(f"Unique Subcategories: {', '.join(subcategories)}\n")




specific_category = "Cyber Attack/ Dependent Crimes"
unique_subcategories = train_df[train_df['category'] == specific_category]['sub_category'].unique()

print(f"Category: {specific_category}")
print(f"Unique Subcategories: {', '.join(unique_subcategories)}")


#########################################################################################


test_df.loc[test_df['category'] == 'Crime Against Women & Children', ['category', 'sub_category']] = ['Cyber Attack/ Dependent Crimes', 'Data Breach/Theft']

# Verify the changes
print(test_df[test_df['category'] == 'Cyber Attack/ Dependent Crimes'])


specific_category = "Cyber Attack/ Dependent Crimes"
unique_subcategories = test_df[test_df['category'] == specific_category]['sub_category'].unique()

print(f"Category: {specific_category}")
print(f"Unique Subcategories: {', '.join(unique_subcategories)}")

test_df.to_csv('Test_cleaned.csv', index=False)

################################################################################################





train_df.loc[train_df['category'] == 'Report Unlawful Content', ['category', 'sub_category']] = ['Any Other Cyber Crime', 'Other']

print(train_df[train_df['category'] == 'Any Other Cyber Crime'])



specific_category = "Any Other Cyber Crime"
unique_subcategories = train_df[train_df['category'] == specific_category]['sub_category'].unique()

print(f"Category: {specific_category}")
print(f"Unique Subcategories: {', '.join(unique_subcategories)}")


train_df.to_csv('Train_cleaned.csv', index=False)


###############################################################################################



