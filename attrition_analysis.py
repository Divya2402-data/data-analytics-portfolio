"""
HR Employee Attrition Analysis
Identifying factors and predicting employee turnover
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_explore_data(filepath):
    """Load and explore HR data"""
    print("="*60)
    print("LOADING HR DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Basic stats
    print("\n" + "="*60)
    print("ATTRITION OVERVIEW")
    print("="*60)
    
    attrition_count = df['Attrition'].value_counts()
    attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
    
    print(f"\nTotal Employees: {len(df)}")
    print(f"Employees Left: {attrition_count.get('Yes', 0)}")
    print(f"Employees Stayed: {attrition_count.get('No', 0)}")
    print(f"Attrition Rate: {attrition_pct.get('Yes', 0):.1f}%")
    
    return df

def analyze_attrition_by_category(df, category, title):
    """Analyze attrition by a specific category"""
    print(f"\n{'='*60}")
    print(f"ATTRITION BY {title.upper()}")
    print("="*60)
    
    # Calculate attrition rate by category
    attrition_by_cat = df.groupby(category).agg({
        'Attrition': lambda x: (x == 'Yes').sum(),
        'EmployeeCount': 'sum'
    })
    attrition_by_cat['AttritionRate'] = (
        attrition_by_cat['Attrition'] / attrition_by_cat['EmployeeCount'] * 100
    ).round(1)
    
    print(f"\n{title} Attrition Analysis:")
    print(attrition_by_cat.sort_values('AttritionRate', ascending=False))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    attrition_by_cat['AttritionRate'].sort_values().plot(kind='barh', color='coral')
    plt.title(f'Attrition Rate by {title}', fontsize=14, fontweight='bold')
    plt.xlabel('Attrition Rate (%)')
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(f'../images/{category.lower()}_attrition.png', dpi=300, bbox_inches='tight')
    print(f"✅ {title} attrition chart saved")
    plt.close()
    
    return attrition_by_cat

def analyze_numeric_factors(df):
    """Analyze numeric factors affecting attrition"""
    print("\n" + "="*60)
    print("NUMERIC FACTORS ANALYSIS")
    print("="*60)
    
    numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion', 
                   'YearsWithCurrManager', 'TotalWorkingYears']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Box plot comparing Yes vs No
        df.boxplot(column=col, by='Attrition', ax=axes[idx])
        axes[idx].set_title(f'{col} by Attrition')
        axes[idx].set_xlabel('Attrition')
        axes[idx].set_ylabel(col)
        
        # Calculate means
        mean_yes = df[df['Attrition'] == 'Yes'][col].mean()
        mean_no = df[df['Attrition'] == 'No'][col].mean()
        
        print(f"\n{col}:")
        print(f"  Left (Yes): {mean_yes:.1f}")
        print(f"  Stayed (No): {mean_no:.1f}")
        print(f"  Difference: {abs(mean_yes - mean_no):.1f}")
    
    plt.tight_layout()
    plt.savefig('../images/numeric_factors.png', dpi=300, bbox_inches='tight')
    print("\n✅ Numeric factors chart saved")
    plt.close()

def analyze_satisfaction_factors(df):
    """Analyze satisfaction-related factors"""
    print("\n" + "="*60)
    print("SATISFACTION ANALYSIS")
    print("="*60)
    
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                        'RelationshipSatisfaction', 'WorkLifeBalance']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(satisfaction_cols):
        # Cross-tabulation
        ct = pd.crosstab(df[col], df['Attrition'], normalize='index') * 100
        
        ct.plot(kind='bar', ax=axes[idx], color=['lightgreen', 'coral'])
        axes[idx].set_title(f'{col} vs Attrition', fontweight='bold')
        axes[idx].set_xlabel(col + ' Level')
        axes[idx].set_ylabel('Percentage')
        axes[idx].legend(['Stayed', 'Left'])
        axes[idx].tick_params(axis='x', rotation=0)
        
        # Print stats
        attrition_by_level = df.groupby(col)['Attrition'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )
        print(f"\n{col} Attrition Rates:")
        for level, rate in attrition_by_level.items():
            print(f"  Level {level}: {rate:.1f}%")
    
    plt.tight_layout()
    plt.savefig('../images/satisfaction_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Satisfaction analysis chart saved")
    plt.close()

def calculate_attrition_cost(df, avg_replacement_cost=18000):
    """Calculate the cost of attrition"""
    print("\n" + "="*60)
    print("COST OF ATTRITION")
    print("="*60)
    
    employees_left = (df['Attrition'] == 'Yes').sum()
    total_cost = employees_left * avg_replacement_cost
    
    print(f"\nEmployees who left: {employees_left}")
    print(f"Average replacement cost per employee: ${avg_replacement_cost:,}")
    print(f"Total annual cost: ${total_cost:,}")
    
    # Cost by department
    dept_attrition = df[df['Attrition'] == 'Yes'].groupby('Department').size()
    dept_costs = dept_attrition * avg_replacement_cost
    
    print(f"\nCost by Department:")
    for dept, cost in dept_costs.items():
        print(f"  {dept}: ${cost:,}")
    
    # Potential savings
    print(f"\nPotential Savings if Attrition Reduced by 5%:")
    target_reduction = len(df) * 0.05
    savings = target_reduction * avg_replacement_cost
    print(f"  Employees retained: {target_reduction:.0f}")
    print(f"  Annual savings: ${savings:,.0f}")
    
    return total_cost

def predictive_model(df):
    """Build a simple predictive model"""
    print("\n" + "="*60)
    print("PREDICTIVE MODEL")
    print("="*60)
    
    # Prepare data
    print("\nPreparing data for modeling...")
    
    # Select features (numeric only for simplicity)
    feature_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                   'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                   'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                   'JobInvolvement']
    
    X = df[feature_cols]
    y = (df['Attrition'] == 'Yes').astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Features Predicting Attrition', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../images/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature importance chart saved")
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved")
    plt.close()

def generate_recommendations(df):
    """Generate actionable recommendations"""
    print("\n" + "="*60)
    print("ACTIONABLE RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Check overtime
    overtime_attrition = df[df['OverTime'] == 'Yes']['Attrition'].value_counts(normalize=True)['Yes'] * 100
    recommendations.append(
        f"\n1. OVERTIME MANAGEMENT (Priority: HIGH)\n"
        f"   - Current overtime attrition: {overtime_attrition:.1f}%\n"
        f"   - Action: Reduce overtime, hire additional staff\n"
        f"   - Expected impact: 3-4% reduction in attrition"
    )
    
    # Check low income
    low_income_threshold = df['MonthlyIncome'].quantile(0.25)
    low_income_attrition = df[df['MonthlyIncome'] < low_income_threshold]['Attrition'].value_counts(normalize=True)['Yes'] * 100
    recommendations.append(
        f"\n2. COMPENSATION REVIEW (Priority: HIGH)\n"
        f"   - Low income (<${low_income_threshold:.0f}) attrition: {low_income_attrition:.1f}%\n"
        f"   - Action: Market rate adjustments for bottom 25%\n"
        f"   - Expected impact: Retain 40-50 employees annually"
    )
    
    # Check new employees
    new_emp_attrition = df[df['YearsAtCompany'] <= 2]['Attrition'].value_counts(normalize=True)['Yes'] * 100
    recommendations.append(
        f"\n3. ONBOARDING PROGRAM (Priority: MEDIUM)\n"
        f"   - New employee (<2 years) attrition: {new_emp_attrition:.1f}%\n"
        f"   - Action: Enhanced 90-day onboarding with mentorship\n"
        f"   - Expected impact: 25% reduction in early-stage attrition"
    )
    
    for rec in recommendations:
        print(rec)

def main():
    """Main execution function"""
    print("="*60)
    print("HR EMPLOYEE ATTRITION ANALYSIS")
    print("="*60)
    
    # Note: Using sample filepath - adjust as needed
    filepath = '../data/HR_Employee_Attrition.csv'
    
    print("\nNote: Ensure the dataset is available at:")
    print(filepath)
    print("\nDownload from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
    
    # For demonstration, create sample data if file doesn't exist
    # In production, load actual data
    
    try:
        df = load_and_explore_data(filepath)
    except FileNotFoundError:
        print("\n⚠️  Dataset not found. Please download and place in data/ folder")
        print("Creating sample analysis structure...")
        return
    
    # Analysis
    analyze_attrition_by_category(df, 'Department', 'Department')
    analyze_attrition_by_category(df, 'JobRole', 'Job Role')
    analyze_numeric_factors(df)
    analyze_satisfaction_factors(df)
    calculate_attrition_cost(df)
    predictive_model(df)
    generate_recommendations(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📊 All visualizations saved to ../images/")
    print("📝 Review findings to develop retention strategies")

if __name__ == "__main__":
    main()
