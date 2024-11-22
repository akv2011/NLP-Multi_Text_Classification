import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import os


warnings.filterwarnings('ignore')


plt.style.use('fivethirtyeight')

def setup_nltk():
    
   
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        try:
            os.makedirs(nltk_data_dir)
        except Exception as e:
            print(f"Warning: Could not create NLTK data directory: {str(e)}")
    
 
    nltk.data.path.append(nltk_data_dir)
    
    
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    
   
    for resource, path in resources.items():
        try:
            try:
                nltk.data.find(path)
                print(f"Resource {resource} already downloaded")
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
                print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Warning: Failed to download/verify {resource}: {str(e)}")
            print(f"Some functionality may be limited without {resource}")


class CybercrimeEDA:
    def __init__(self, df):
        self.df = df
    
        setup_nltk()
        self.prepare_data()
        
    def prepare_data(self):
        
        try:
            
            self.df = self.df.copy()
            
            
            self.df['text_length'] = self.df['crimeaditionalinfo'].fillna('').str.len()
            self.df['word_count'] = self.df['crimeaditionalinfo'].fillna('').str.split().str.len()
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    def text_analysis(self):
    
        try:
            
            all_text = ' '.join(self.df['crimeaditionalinfo'].fillna('').astype(str))
            
            try:
                tokens = word_tokenize(all_text.lower())
            except LookupError:
                print("Warning: NLTK punkt tokenizer not available. Using basic split()")
                tokens = all_text.lower().split()
            
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                print("Warning: Stopwords not available. Using basic English stopwords")
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
            
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            
      
            word_freq = Counter(tokens)
            
            
            plt.figure(figsize=(15, 6))
            word_freq_df = pd.DataFrame.from_dict(
                dict(word_freq.most_common(20)), 
                orient='index', 
                columns=['count']
            )
            sns.barplot(x=word_freq_df.index, y='count', data=word_freq_df)
            plt.title('Top 20 Most Common Words in Crime Descriptions')
            plt.xlabel('Word')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            

            try:
                wordcloud = WordCloud(
                    width=1600,
                    height=800,
                    background_color='white'
                ).generate(' '.join(tokens))
                
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud of Crime Descriptions')
                plt.show()
            except Exception as e:
                print(f"Warning: Could not generate word cloud: {str(e)}")
            
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            print("Skipping text analysis visualizations")

    def plot_category_distribution(self):
        
        plt.figure(figsize=(15, 8))
        
       
        category_counts = self.df['category'].value_counts()
        
        
        sns.barplot(
            x=category_counts.values,
            y=category_counts.index,
            palette='viridis'
        )
        plt.title('Distribution of Crime Categories')
        plt.xlabel('Count')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.show()
        
        
        plt.figure(figsize=(15, 8))
        
       
        subcategory_counts = self.df['sub_category'].value_counts().head(15)
        
    
        sns.barplot(
            x=subcategory_counts.values,
            y=subcategory_counts.index,
            palette='viridis'
        )
        plt.title('Top 15 Sub-Categories')
        plt.xlabel('Count')
        plt.ylabel('Sub-Category')
        plt.tight_layout()
        plt.show()
        
    def category_subcategory_analysis(self):
        
        pivot_table = pd.crosstab(
            self.df['category'],
            self.df['sub_category']
        )
        
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            pivot_table,
            cmap='YlOrRd',
            annot=True,
            fmt='d',
            cbar_kws={'label': 'Count'}
        )
        plt.title('Category vs Sub-Category Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def basic_analysis(self):
       
        print("Dataset Overview:")
        print("-" * 50)
        print(f"Total number of records: {len(self.df):,}")
        print(f"Number of categories: {self.df['category'].nunique()}")
        print(f"Number of sub-categories: {self.df['sub_category'].nunique()}")
        
       
        print("\nMissing Values:")
        print("-" * 50)
        print(self.df.isnull().sum())
        
     
        print("\nText Statistics:")
        print("-" * 50)
        print(self.df[['text_length', 'word_count']].describe())
        
    def text_length_analysis(self):
       
        plt.figure(figsize=(15, 6))
        sns.boxplot(
            data=self.df,
            x='category',
            y='text_length',
            palette='viridis'
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Text Length by Category')
        plt.xlabel('Category')
        plt.ylabel('Text Length')
        plt.tight_layout()
        plt.show()
        
    def generate_insights(self):
       
        insights = []
        
        
        main_category = self.df['category'].mode()[0]
        category_ratio = (self.df['category'].value_counts().iloc[0] / 
                         len(self.df) * 100)
        
        insights.append(f"Most common crime category: {main_category} "
                       f"({category_ratio:.1f}% of all cases)")
        
        
        avg_text_length = self.df['text_length'].mean()
        insights.append(f"Average description length: {avg_text_length:.1f} characters")
        
        
        category_complexity = self.df.groupby('category')['text_length'].mean()
        most_detailed = category_complexity.idxmax()
        insights.append(f"Category with most detailed descriptions: {most_detailed}")
        
        return insights

def run_complete_eda(data_path):
    
    try:
       
        df = pd.read_csv(data_path)
        
       
        eda = CybercrimeEDA(df)
        
        # Run analyses with error handling
        print("Running Basic Analysis...")
        eda.basic_analysis()
        
        print("\nGenerating Category Distribution Plots...")
        eda.plot_category_distribution()
        
        print("\nAnalyzing Category-Subcategory Relationships...")
        eda.category_subcategory_analysis()
        
        print("\nPerforming Text Analysis...")
        eda.text_analysis()
        
        print("\nAnalyzing Text Length Distribution...")
        eda.text_length_analysis()
        
        print("\nKey Insights:")
        for i, insight in enumerate(eda.generate_insights(), 1):
            print(f"{i}. {insight}")
            
        return eda
        
    except Exception as e:
        print(f"Error running EDA: {str(e)}")
        raise




        



data_path = "Train_cleaned.csv"
eda = run_complete_eda(data_path)
