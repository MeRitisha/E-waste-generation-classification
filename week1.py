import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
import requests
import io
import warnings
warnings.filterwarnings('ignore')

class EWasteClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.best_model = None
        self.feature_names = []
        
    def load_data_from_github(self, url=None):
        """
        Load data from GitHub repository or use sample dataset
        Supports both CSV files and creates sample data if URL not accessible
        """
        try:
            if url:
                # Try to load from GitHub URL
                response = requests.get(url)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.text))
                    print(f"Successfully loaded data from GitHub: {df.shape}")
                    return df
                else:
                    print(f"Could not access URL. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error loading from GitHub: {e}")
        
        # Create realistic e-waste dataset based on common patterns
        print("Creating sample e-waste classification dataset...")
        return self.create_sample_ewaste_dataset()
    
    def create_sample_ewaste_dataset(self, n_samples=2000):
        """Create a realistic e-waste classification dataset"""
        np.random.seed(42)
        
        # E-waste categories commonly found in datasets
        categories = ['Mobile Phone', 'Laptop', 'Desktop Computer', 'Television', 
                     'Tablet', 'Printer', 'Camera', 'Gaming Console', 'Monitor', 'Router']
        
        data = []
        
        for i in range(n_samples):
            category = np.random.choice(categories)
            
            # Generate features based on category
            if category == 'Mobile Phone':
                weight = np.random.normal(0.15, 0.05)  # kg
                screen_size = np.random.normal(6.2, 1.0)  # inches
                battery_capacity = np.random.normal(3500, 800)  # mAh
                age = np.random.exponential(3)  # years
                condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                           p=[0.2, 0.3, 0.3, 0.2])
                
            elif category == 'Laptop':
                weight = np.random.normal(2.2, 0.8)
                screen_size = np.random.normal(14, 2)
                battery_capacity = np.random.normal(5000, 1500)
                age = np.random.exponential(4)
                condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                           p=[0.15, 0.25, 0.35, 0.25])
                
            elif category == 'Desktop Computer':
                weight = np.random.normal(8.5, 3.0)
                screen_size = 0  # No built-in screen
                battery_capacity = 0  # No battery
                age = np.random.exponential(5)
                condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                           p=[0.1, 0.2, 0.4, 0.3])
                
            elif category == 'Television':
                weight = np.random.normal(15.0, 8.0)
                screen_size = np.random.normal(42, 15)
                battery_capacity = 0
                age = np.random.exponential(7)
                condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                           p=[0.1, 0.3, 0.4, 0.2])
                
            else:  # Other categories
                weight = np.random.normal(1.5, 1.0)
                screen_size = np.random.normal(10, 5)
                battery_capacity = np.random.normal(2000, 1000)
                age = np.random.exponential(4)
                condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                           p=[0.15, 0.3, 0.35, 0.2])
            
            # Ensure positive values
            weight = max(0.1, weight)
            screen_size = max(0, screen_size)
            battery_capacity = max(0, battery_capacity)
            age = max(0.1, age)
            
            # Additional features
            brand_tier = np.random.choice(['Premium', 'Mid-range', 'Budget'], 
                                        p=[0.3, 0.4, 0.3])
            has_damage = np.random.choice([0, 1], p=[0.7, 0.3])
            recyclable_materials = np.random.normal(0.8, 0.15)
            recyclable_materials = max(0, min(1, recyclable_materials))
            
            # Calculate hazardous score based on category and features
            hazard_base = {
                'Mobile Phone': 0.3, 'Laptop': 0.4, 'Desktop Computer': 0.5,
                'Television': 0.6, 'Tablet': 0.25, 'Printer': 0.45,
                'Camera': 0.2, 'Gaming Console': 0.35, 'Monitor': 0.5, 'Router': 0.3
            }
            
            hazardous_score = hazard_base[category] + np.random.normal(0, 0.1)
            hazardous_score = max(0, min(1, hazardous_score))
            
            data.append({
                'category': category,
                'weight_kg': weight,
                'screen_size_inches': screen_size,
                'battery_capacity_mah': battery_capacity,
                'age_years': age,
                'condition': condition,
                'brand_tier': brand_tier,
                'has_damage': has_damage,
                'recyclable_materials_ratio': recyclable_materials,
                'hazardous_score': hazardous_score
            })
        
        df = pd.DataFrame(data)
        
        # Create disposal method target based on features
        def determine_disposal_method(row):
            if row['condition'] in ['Excellent', 'Good'] and row['age_years'] < 3:
                return 'Reuse'
            elif row['recyclable_materials_ratio'] > 0.7 and row['hazardous_score'] < 0.4:
                return 'Recycle'
            elif row['hazardous_score'] > 0.6:
                return 'Hazardous Disposal'
            else:
                return 'Standard Disposal'
        
        df['disposal_method'] = df.apply(determine_disposal_method, axis=1)
        
        return df
    
    def load_and_preprocess_data(self, df, target_column='disposal_method'):
        """Preprocess the data for machine learning"""
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Target column: {target_column}")
        
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found.")
            print("Available columns:", df.columns.tolist())
            # Use the last column as target if specified target not found
            target_column = df.columns[-1]
            print(f"Using '{target_column}' as target column.")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        print(f"Categorical columns: {categorical_columns.tolist()}")
        print(f"Numerical columns: {numerical_columns.tolist()}")
        
        # Process categorical variables
        X_processed = pd.DataFrame()
        
        # One-hot encode categorical variables
        for col in categorical_columns:
            dummies = pd.get_dummies(X[col], prefix=col)
            X_processed = pd.concat([X_processed, dummies], axis=1)
        
        # Add numerical variables
        for col in numerical_columns:
            X_processed[col] = X[col]
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
        
        self.feature_names = X_processed.columns.tolist()
        
        print(f"Processed features shape: {X_scaled.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X_scaled, y_encoded, X_processed.columns.tolist()
    
    def exploratory_data_analysis(self, df, target_column='disposal_method'):
        """Perform comprehensive EDA on the dataset"""
        print("=== E-WASTE CLASSIFICATION - EXPLORATORY DATA ANALYSIS ===\n")
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print("\nDataset Info:")
        print(df.info())
        
        # Target distribution
        if target_column in df.columns:
            print(f"\n{target_column} distribution:")
            target_counts = df[target_column].value_counts()
            print(target_counts)
            print(f"\nTarget percentages:")
            print((target_counts / len(df) * 100).round(2))
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Target distribution
        if target_column in df.columns:
            df[target_column].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title(f'{target_column} Distribution')
            axes[0,0].set_xlabel(target_column)
            axes[0,0].set_ylabel('Count')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Category distribution
        if 'category' in df.columns:
            df['category'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
            axes[0,1].set_title('E-Waste Category Distribution')
            axes[0,1].set_xlabel('Category')
            axes[0,1].set_ylabel('Count')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Numerical features distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            df[numerical_cols[0]].hist(bins=30, ax=axes[1,0], alpha=0.7, color='orange')
            axes[1,0].set_title(f'{numerical_cols[0]} Distribution')
            axes[1,0].set_xlabel(numerical_cols[0])
            
            df[numerical_cols[1]].hist(bins=30, ax=axes[1,1], alpha=0.7, color='purple')
            axes[1,1].set_title(f'{numerical_cols[1]} Distribution')
            axes[1,1].set_xlabel(numerical_cols[1])
        
        # Correlation heatmap for numerical features
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[2,0])
            axes[2,0].set_title('Numerical Features Correlation')
        
        # Box plot for age vs disposal method
        if 'age_years' in df.columns and target_column in df.columns:
            df.boxplot(column='age_years', by=target_column, ax=axes[2,1])
            axes[2,1].set_title('Age Distribution by Disposal Method')
            axes[2,1].set_xlabel('Disposal Method')
            axes[2,1].set_ylabel('Age (years)')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance analysis using Random Forest
        if target_column in df.columns:
            # Prepare data for feature importance
            X_temp = df.drop(target_column, axis=1)
            y_temp = df[target_column]
            
            # Handle categorical variables for quick analysis
            X_encoded = pd.get_dummies(X_temp)
            y_encoded = self.label_encoder.fit_transform(y_temp)
            
            # Quick Random Forest for feature importance
            rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_temp.fit(X_encoded, y_encoded)
            
            feature_importance = pd.DataFrame({
                'feature': X_encoded.columns,
                'importance': rf_temp.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
            return feature_importance
        
        return None
    
    def initialize_models(self):
        """Initialize different ML models with optimized parameters"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', 
                random_state=42, 
                class_weight='balanced', 
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            )
        }
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models with comprehensive metrics"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        print("=== MODEL TRAINING AND EVALUATION ===\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted'
                )
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'y_test': y_test
                }
                
                print(f"{name} Results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print()
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return results, X_test, y_test
    
    def detailed_evaluation(self, results, X_test, y_test):
        """Provide detailed evaluation of all models"""
        if not results:
            print("No models to evaluate.")
            return None, None
        
        # Find best model based on F1-score
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_model = results[best_model_name]['model']
        self.best_model = best_model
        
        print(f"=== DETAILED EVALUATION - BEST MODEL: {best_model_name} ===\n")
        
        y_pred = results[best_model_name]['predictions']
        
        # Classification report
        target_names = self.label_encoder.classes_
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
        comparison_data = []
        
        for model_name in results.keys():
            row = [model_name]
            for metric in metrics:
                row.append(results[model_name][metric])
            comparison_data.append(row)
        
        model_comparison = pd.DataFrame(
            comparison_data, 
            columns=['Model'] + [m.replace('_', ' ').title() for m in metrics]
        ).sort_values('F1 Score', ascending=False)
        
        print("\nModel Comparison:")
        print(model_comparison.to_string(index=False, float_format='%.4f'))
        
        # Visualization
        plt.figure(figsize=(14, 8))
        
        # Performance comparison
        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(model_comparison))
        plt.bar(x_pos, model_comparison['F1 Score'], alpha=0.7, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('Model Performance Comparison (F1 Score)')
        plt.xticks(x_pos, model_comparison['Model'], rotation=45)
        
        # Feature importance for best model
        plt.subplot(1, 2, 2)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Features - {best_model_name}')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name, best_model
    
    def predict_new_samples(self, features_dict):
        """Make predictions on new samples"""
        if self.best_model is None:
            print("No trained model available. Please train the model first.")
            return None
        
        try:
            # Convert to DataFrame
            new_data = pd.DataFrame([features_dict])
            
            # Process categorical variables to match training data
            processed_data = pd.DataFrame()
            
            for feature in self.feature_names:
                if feature in new_data.columns:
                    processed_data[feature] = new_data[feature]
                else:
                    # Handle one-hot encoded features
                    processed_data[feature] = 0
                    for col in new_data.columns:
                        if feature.startswith(col + '_'):
                            value = feature.split('_', 1)[1]
                            if new_data[col].iloc[0] == value:
                                processed_data[feature] = 1
            
            # Scale the data
            processed_scaled = self.scaler.transform(processed_data)
            
            # Make prediction
            prediction = self.best_model.predict(processed_scaled)
            prediction_proba = self.best_model.predict_proba(processed_scaled)
            
            predicted_class = self.label_encoder.inverse_transform(prediction)[0]
            
            print(f"Predicted E-Waste Disposal Method: {predicted_class}")
            print("Prediction Probabilities:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"  {class_name}: {prediction_proba[0][i]:.4f}")
            
            return predicted_class, prediction_proba[0]
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None

def main():
    """Main execution function"""
    # Initialize classifier
    classifier = EWasteClassifier()
    
    # Load data - you can specify the GitHub raw file URL here
    # Example: https://raw.githubusercontent.com/dularibhatt/AICTE-Internship/main/P3%20-%20E-Waste%20Generation%20Classification/dataset.csv
    github_url = "https://raw.githubusercontent.com/dularibhatt/AICTE-Internship/main/P3%20-%20E-Waste%20Generation%20Classification/dataset.csv"# Replace with actual dataset URL
    
    print("Loading e-waste dataset...")
    df = classifier.load_data_from_github(github_url)
    

    print("\nDataset preview:")
    print(df.head())
    print("\nDataset description:")
    print(df.describe())
    
    # Perform EDA
    feature_importance = classifier.exploratory_data_analysis(df)
    

    target_column = 'disposal_method'  
    X, y, feature_names = classifier.load_and_preprocess_data(df, target_column)
    
    
    classifier.initialize_models()
    results, X_test, y_test = classifier.train_and_evaluate_models(X, y)
    
    
    if results:
        best_model_name, best_model = classifier.detailed_evaluation(results, X_test, y_test)
        
        
        print("\n=== EXAMPLE PREDICTION ===")
        sample_features = {
            'category': 'Mobile Phone',
            'weight_kg': 0.18,
            'screen_size_inches': 6.1,
            'battery_capacity_mah': 3200,
            'age_years': 2.5,
            'condition': 'Good',
            'brand_tier': 'Premium',
            'has_damage': 0,
            'recyclable_materials_ratio': 0.75,
            'hazardous_score': 0.3
        }
        
        classifier.predict_new_samples(sample_features)
        
        print(f"\n=== PROJECT COMPLETED SUCCESSFULLY ===")
        print(f"Best performing model: {best_model_name}")
        print("You can now use this model to classify e-waste disposal methods!")
    else:
        print("No models were successfully trained.")

if __name__ == "__main__":
    main()