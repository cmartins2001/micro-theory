# DataBricks Notebooks Folder

_I am using this subfolder to store all DataBricks notebooks associated with replication of Autor, et al.'s 2013 paper on Chinese manufacturing shocks to local U.S. labor markets. In this readME, I also track some of the lessons I learn about how DataBricks works and how it can serve my replication/extension purposes._

### Basic ETL Process:
1. I load `.dta` files from David Dorn's personal website that are stored on my local machine to a Volume in DataBricks community edition. I do this entirely through the UI.
2. I use the `stata_to_delta` notebook in this folder to read that data into a `pandas` dataframe, which I convert to a Spark dataframe(s).
3. In that same notebook, I can conduct profiling and any necessary cleaning before saving the Spark dataframe(s) as "silver" delta tables. _There likely doesn't need to be a gold layer since these tables are already so clean, and I suppose the `.dta` files are my "bronze" layer. This is simpler than a more robust ingestion/transformation process but my goal is to build my understanding of the workflow._

### Notes on the Process:
The delta tables I create in Step 3 are technically stored in a DataBricks-managed cloud location (likely an AWS S3) that I don't have direct access too. Once the tables are created, I can use the serverless SQL Warehouse provided by DataBricks to interact with them by writing SQL code in a DataBricks notebook or by using DataBricks SQL. 

Also, since the tables are created within my catalog, Genie knows the data and I can literally write natural language prompts that it quickly converts into SQL queries. I can also rapidly develop dashboards and visuals from the delta tables.
