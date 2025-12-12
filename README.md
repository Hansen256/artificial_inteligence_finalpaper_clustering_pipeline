# Customer Segmentation Using Clustering

A machine learning project for segmenting e-commerce customers using clustering algorithms (K-Means and DBSCAN) to enable targeted marketing strategies.

## Objective

Build a clustering pipeline to segment customers from a Nigerian e-commerce dataset based on Recency and Frequency metrics, enabling data-driven marketing recommendations and customer targeting strategies.

## Dataset

**Source:** [Nigerian E-Commerce Sales Dataset](https://www.kaggle.com/datasets/babajidedairo/nigerian-ecommerce-sales-dataset)

**Time Period:** February - May 2021 (4 months)

**Key Features:**

- 23 unique customer businesses
- Transaction data including order dates, quantities, and total prices
- Order IDs and business names for customer identification

## Project Structure

```txt
final_clustering/
├── clustering_ecommerce.ipynb      # Main Jupyter notebook with full analysis
├── customer_segments.csv            # Output file with customer cluster assignments
└── README.md                        # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook or JupyterLab

### Clone the Repository

```bash
git clone https://github.com/Hansen256/artificial_inteligence_finalpaper_clustering_pipeline.git
cd artificial_inteligence_finalpaper_clustering_pipeline/final_clustering
```

### Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub openpyxl
```

Or install via requirements file (if available):

```bash
pip install -r requirements.txt
```

## Running the Project

1. Navigate to the project directory:

   ```bash
   cd final_clustering
   ```

2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook clustering_ecommerce.ipynb
   ```

3. Execute all cells sequentially to:
   - Load and clean the dataset
   - Engineer Recency and Frequency features
   - Run K-Means clustering (k=2 to k=10)
   - Apply PCA for 2D visualization
   - Compare with DBSCAN algorithm
   - Generate cluster profiles and marketing recommendations

## Project Components

### Part A: Data Preparation (10 Marks)

- Load dataset and display structure
- Remove duplicates and handle missing values
- Engineer Recency (days since last purchase) and Frequency (transaction count) features
- Scale numerical features using StandardScaler

### Part B: Dense NN / K-Means Clustering (15 Marks)

- Run K-Means for k=2 to k=10
- Calculate inertia and silhouette scores
- Identify optimal k using elbow method and silhouette analysis
- Apply PCA for 2D visualization
- Compare with DBSCAN density-based clustering

### Part C: Evaluation (15 Marks)

- Compute silhouette scores for cluster quality assessment
- Create cluster profile tables with mean values
- Generate marketing recommendations per segment
- Save final cluster assignments to CSV

## Key Results

### Optimal K Selection

The model selects the optimal number of clusters based on:

1. Elbow method (inertia reduction rate)
2. Silhouette score (cluster separation quality)

### Cluster Segments & Recommendations

| Segment | Characteristics | Marketing Action |
|---------|-----------------|-----------------|
| Active High-Value | Recent, High frequency | VIP programs and exclusive early access |
| At-Risk | Not recent, Was frequent | Win-back campaigns with special discounts |
| New Customers | Recent, Low frequency | Welcome series and second-purchase incentives |
| Dormant | Not recent, Low frequency | Aggressive reactivation with "We miss you" offers |

## Algorithms Used

### K-Means Clustering

- Centroid-based algorithm
- Forces all points into clusters
- Spherical cluster assumption
- Best for marketing segmentation (no noise points)

### DBSCAN Clustering

- Density-based algorithm
- Identifies noise/outlier points
- Discovers arbitrary cluster shapes
- Useful for anomaly detection

## Output Files

### customer_segments.csv

- CustomerID: Business name/customer identifier
- Recency: Days since last purchase
- Frequency: Number of transactions
- Cluster: Assigned cluster number
- Segment: Segment name (Active High-Value, At-Risk, etc.)

## Limitations

1. Limited time period (4 months) - may not capture seasonal patterns
2. Only two features used - could benefit from monetary value and demographic data
3. Static segmentation - requires periodic retraining as customer behavior evolves
4. Small customer base (23 unique customers) - limited statistical power
5. K-Means assumes spherical, equally-sized clusters

## Next Steps

1. Implement automated retraining pipeline for monthly segment updates
2. A/B test marketing recommendations and measure ROI impact
3. Enrich features with product preferences, browsing behavior, and demographics
4. Explore hierarchical clustering for sub-segmentation
5. Build churn prediction and propensity models using cluster membership

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Visualization
- seaborn: Statistical data visualization
- scikit-learn: Machine learning algorithms (KMeans, DBSCAN, PCA, StandardScaler)
- kagglehub: Dataset download from Kaggle

## Author

Tumusiime Hansen Andrew

## References

- Nigerian E-Commerce Sales Dataset: https://www.kaggle.com/datasets/babajidedairo/nigerian-ecommerce-sales-dataset  <!--markdownlint-disable-line-->
- Scikit-learn Documentation: https://scikit-learn.org/ <!--markdownlint-disable-line-->
