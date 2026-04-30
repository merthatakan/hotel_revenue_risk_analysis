import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =================================================================
# 1. INTRODUCTION & STRATEGIC CONTEXT
# =================================================================
print("""
PROJECT: HOTEL RESERVATION CANCELLATION RISK & FINANCIAL IMPACT ANALYSIS

OBJECTIVE:
The primary operational risk for hotels is the "No-Show" phenomenon. 
This analysis examines the impact of 'Lead Time' (the gap between booking 
and arrival) and customer special requests on cancellation rates.

The project quantification demonstrates how wait times and personalization 
affect revenue stability. The dataset is simulated based on the 
Kaggle "Hotel Reservations Classification Dataset" standards.

SOURCE: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset
""")

# =================================================================
# 2. DATASET CREATION & PRE-PROCESSING
# =================================================================
# Simulated dataset representing real-world hotel booking scenarios
data = {
    'Booking_ID': range(101, 111),
    'Lead_Time': [15, 45, 2, 90, 5, 20, 120, 10, 60, 3],
    'Room_Type': ['Standard', 'Deluxe ', 'Standard', 'Suite', 'Standard', 'Deluxe', 'Suite', 'Standard', 'Deluxe', 'Standard'],
    'Avg_Price': [850, 1450, 780, 2600, 900, 1300, 3100, 820, 1550, 880],
    'Status': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0], # 0: Confirmed, 1: Canceled
    'Special_Requests': [1, 0, 2, 0, 1, 2, 0, 1, 0, 3]
}
df = pd.DataFrame(data)

# Data Cleaning
df['Room_Type'] = df['Room_Type'].str.strip() 
df['Avg_Price'] = df['Avg_Price'].astype(int)

# =================================================================
# 3. STATISTICAL CALCULATIONS & KPI ANALYSIS
# =================================================================
# Calculating a dynamic risk threshold using Mean + Standard Deviation
lead_time_array = df['Lead_Time'].to_numpy()
risk_threshold = np.mean(lead_time_array) + np.std(lead_time_array)

# Financial KPIs 
realized_revenue = df[df['Status'] == 0]['Avg_Price'].sum()
lost_revenue = df[df['Status'] == 1]['Avg_Price'].sum()
total_potential_revenue = realized_revenue + lost_revenue

# Segmentation: Categorizing lead times into strategic risk groups
df['Time_Group'] = pd.cut(df['Lead_Time'], 
                          bins=[0, 20, 60, 150], 
                          labels=['0-20 Days (Safe)', '21-60 Days (Risky)', '60+ Days (Critical)'])

# =================================================================
# 4. PROFESSIONAL REPORTING MODULE
# =================================================================
print("\n" + "="*100)
print(f"{'PROCESSED HOTEL RESERVATION DATASET':^100}")
print("="*100)
print(df.to_string(index=False, col_space=18)) 

print("\n" + "-"*80)
print(f"{'FINANCIAL PERFORMANCE SUMMARY':^80}")
print("-"*80)
print(f"{'Total Potential Revenue':<40} | {total_potential_revenue:>30} $")
print(f"{'Realized Revenue':<40} | {realized_revenue:>30} $")
print(f"{'Cancellation Loss (Opportunity Cost)':<40} | {lost_revenue:>30} $")
print(f"{'Revenue Loss Percentage':<40} | %{(lost_revenue/total_potential_revenue*100):>31.1f}")
print("-"*80)

# =================================================================
# 5. VISUALIZATION LAYER 
# =================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#f8f9fa')
fig.suptitle('Hotel Operations: Strategic Risk & Revenue Dashboard', 
             fontsize=22, fontweight='bold', color='#2c3e50', y=0.98)

# --- GRAPH 1: Scatter Plot (Moved further left to avoid text overlap) ---
sns.scatterplot(data=df, x='Lead_Time', y='Avg_Price', hue='Status', 
                size='Avg_Price', sizes=(80, 350), palette='RdYlGn_r', 
                ax=axes[0, 0], edgecolor='white', alpha=0.9)

axes[0, 0].axvline(risk_threshold, color='#e74c3c', linestyle='--', linewidth=2.5, 
                   label=f'Risk Threshold ({int(risk_threshold)} Days)')


axes[0, 0].legend(bbox_to_anchor=(-0.25, 1), loc='upper right', borderaxespad=0., 
                  title="Status & Price", title_fontsize='9', shadow=True, 
                  prop={'size': 7.5}, labelspacing=0.4)

axes[0, 0].set_title('Lead Time vs. Price Risk Distribution', fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Lead Time (Days)', fontsize=11)
axes[0, 0].set_ylabel('Average Price ($)', fontsize=11)

# --- GRAPH 2: Bar Chart ---
sns.barplot(data=df.groupby('Room_Type')['Avg_Price'].sum().reset_index(), 
            x='Room_Type', y='Avg_Price', ax=axes[0, 1], palette='viridis', edgecolor='.2')
axes[0, 1].set_title('Total Revenue by Room Category', fontsize=16, fontweight='bold', pad=15)

# --- GRAPH 3: Pie Chart ---
axes[1, 0].pie([realized_revenue, lost_revenue], labels=['Realized Revenue', 'Opportunity Loss'], 
               autopct='%1.1f%%', colors=['#66bb6a', '#ef5350'], 
               startangle=140, explode=[0, 0.1], shadow=False,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 11})
axes[1, 0].set_title('Revenue vs. Opportunity Loss Analysis', fontsize=16, fontweight='bold', pad=25)

# --- GRAPH 4: Box Plot ---
sns.boxplot(data=df, x='Time_Group', y='Avg_Price', hue='Status', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Price Range & Cancellation by Segment', fontsize=16, fontweight='bold', pad=15)
axes[1, 1].legend(loc='upper right', title="Status", prop={'size': 8})


plt.subplots_adjust(left=0.2, wspace=0.35, hspace=0.4)


fig.text(0.95, 0.02, 'Analysis by: Mert Hüseyin Atakan', 
         fontsize=11, style='italic', fontweight='bold', ha='right', color='#2c3e50')

plt.savefig('hotel_dashboard_final_v3.png', dpi=300, bbox_inches='tight')
plt.show()