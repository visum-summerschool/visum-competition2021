# %%
"""
# VISUM 2021 - data inspection

The dataset of the VISUM 2021 project comprises two main `.csv` tables, namely:

### 1. Outfits table

The Outfits table relates every outfit along with the corresponding set of products that belong to it. It is available at the `df_outfits.csv` file and contains 3 columns:
- `outfit_id`: the outfit id;
- `main_product_id`: the main product id, representing the anchor product in the outfit;
- `outfit_products`: the set of product ids that belong to the outfit.

### 2. Products table

The Products table relates every product present in the Outfits' table along with the required product information (i.e. product name, category, and description). It is available at the `df_products.csv` file and contains 4 columns:
- `productid`: the product id;
- `productname`: the product name;
- `category`: the product category;
- `description`: the product description.
"""

"""
## Imports block
"""
import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap
from prettytable import PrettyTable
from tqdm import tqdm

"""
## Global variables block
"""
DATA_DIR = '../master/dataset/train/'
DF_OUTFITS_FN = 'df_outfits.csv'
DF_PRODUCTS_FN = 'df_products.csv'
IMAGES_DIR = os.path.join(DATA_DIR, 'product_images')
SAVE_PATH = 'inspection_images/'
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)
extensions = [".jpg", ".jpeg", ".png"]

"""
## Functions definition block
"""
def from_np_array(array_string):
    ''' Convert a array_string to numpy array
        
    Args:
        array_string (str): array_string.

    Returns:
        (numpy.array): numpy array.
    
    '''
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def display_outfit(outfit_id, df_outfits, df_products, imgs_dir):
    ''' Display all product's information of a given outfit (outfit_id)
        
    Args:
        outfit_id (int): outfit id.
        df_outfits (pandas.dataframe): outfits dataframe.
        df_products (pandas.dataframe): products dataframe.
        imgs_dir (str): products images directory.

    Returns:
        None
    
    '''

    # get outfit products
    outfit = df_outfits[df_outfits['outfit_id'] == outfit_id]['outfit_products'].iloc[0]

    # init outfit table
    outfit_table = PrettyTable()
    outfit_table.field_names = ["Id", "productname", "category", "description"]

    for i, p in enumerate(outfit):
        # read product image
        im_fn = os.path.join(imgs_dir, str(p) + '.jpg')
        im = mpimg.imread(im_fn)

        # get product info
        prod_info = df_products.loc[df_products['productid'] == p].iloc[0]

        # add row to outfit table
        desc = wrap(str(prod_info['description']) or '', 65) or ['']
        name = wrap(str(prod_info['productname']) or '', 15)
        name =  name + [''] * (len(desc)-len(name))
        outfit_table.add_row([prod_info['productid'], name[0], prod_info['category'], desc[0]])
        for subseq_desc, subseq_name in zip(desc[1:], name[1:]):
            outfit_table.add_row(['', subseq_name, '', subseq_desc])

        # imshow
        plt.subplot(1, len(outfit), i+1)
        plt.imshow(im)
        plt.title(str(p))
        plt.axis('off')

    plt.savefig(os.path.join(SAVE_PATH, str(outfit_id)+".png"))
    outfit_table.align = "l"
    print("Outfit ", outfit_id)
    print(outfit_table)

def get_min_max_avg(df):
    '''
        Get max, min and avg values sepecific column in given dataframe
        
        Args:
            df (Dataframe): dataframe (df_outifts)
        
        Return:
            max (int): max value of items in an outfit
            min (int): min value of items in an outfit
            avg (int): avg value of items in an outfit
    '''
    products = []
    for idx, row in df.iterrows():
        outfit = row['outfit_products']
        products.append(len(outfit))
    avg = sum(products) / len(products)
    return np.max(products), np.min(products), round(avg,1)


"""
## Read `df_outfits` and `df_products` table
"""
# Read df_outfits
df_outfits_fn = os.path.join(DATA_DIR, DF_OUTFITS_FN)
df_outfits = pd.read_csv(df_outfits_fn, converters={'outfit_products': from_np_array}, index_col=0)
print("Outfits:\n", df_outfits)

# Read df_products
df_products_fn = os.path.join(DATA_DIR, DF_PRODUCTS_FN)
df_products = pd.read_csv(df_products_fn, index_col=0)
print("Products:\n", df_products)

"""
## Display Outfits info
"""
outfit_id = 1002569
display_outfit(outfit_id, df_outfits, df_products, IMAGES_DIR)

"""
### Dataset statistics
- Min, Max, and AVG number of products per outfit
- Category distribution (total number of categories, table, bar plot)
"""
#Avg/Min/Max number of products per outfit
max_prods, min_prods, avg_prods = get_min_max_avg(df_outfits)
print(f"max prods per outfit: {max_prods}, min prods per outfit: {min_prods}, avg prods per outfit: {avg_prods}")

#Get total of unique categories
print(f"number of categories: {len(df_products['category'].unique())}")

#Get number of unique products
n_products_unique = len(df_products)
print(f"number of unique products: {n_products_unique}")

#Get number of unique outfits
n_unique_outfits = len(df_outfits)
print(f"number of unique outfits: {n_unique_outfits}")

# %%
#create histogram number_of_products by category
products_count = df_products['category'].value_counts()
plt.figure(figsize=(40, 30))
plot = plt.bar(products_count.index, products_count.values)
plt.title('Frequency Distribution of products by categories')
plt.ylabel('Number of Products', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.xticks(rotation=90, fontsize=15)
plt.savefig(os.path.join(SAVE_PATH, "categories.png"))

# %%
# distribution of products per category 
df = pd.DataFrame(data= {'category': list(products_count.index), 'n_products': list(products_count.values)})
pd.set_option('display.max_rows', None)
print(df)
