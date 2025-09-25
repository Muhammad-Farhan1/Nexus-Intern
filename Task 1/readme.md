
# Online Retail Data Analysis

## 📊 Project Overview
This Jupyter Notebook analyzes an online retail dataset to uncover sales trends, customer behavior patterns, and geographical insights. The dataset contains transaction records from an online retail store.

## 📁 Dataset Information
- **File**: `OnlineRetail.csv`
- **Records**: 541,909 entries
- **Columns**: 8 attributes including InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

## 🔧 Data Preprocessing
- Handled missing values in Description (mode) and CustomerID (median)
- Removed 5,268 duplicate rows
- Converted InvoiceDate to datetime format
- Added calculated columns for Month and Total_Amount

## 📈 Analysis Performed

### 1. Monthly Sales Trend
- Created time series analysis of sales by month
- November 2011 showed peak sales (£1.45M)
- Visualized with line plot showing seasonal patterns

### 2. Customer Sales Distribution
- Analyzed total sales per customer
- Histogram revealed most customers have low spending
- Few high-value customers contribute significantly to revenue

### 3. Sales by Country
- United Kingdom dominates (92% of total sales)
- Netherlands, Ireland, Germany next largest markets
- Bar chart visualization of top 10 countries

## 📊 Key Findings
- Strong seasonal pattern with holiday peak
- Pareto distribution in customer spending
- UK-focused business with international potential
- Data quality issues with negative quantities/prices

## 🛠️ Technical Implementation
- **Libraries**: pandas, matplotlib
- **Techniques**: Data cleaning, datetime conversion, GroupBy operations, visualization
- **Visualizations**: Line plots, histograms, bar charts

## 💡 Business Implications
- Seasonal marketing planning around November
- Customer segmentation for high-value clients
- International expansion opportunities
- Inventory alignment with sales patterns
