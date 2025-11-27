import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import argparse
import os
import sys
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# --- Configuration ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['pdf.fonttype'] = 42 

class SalesAnalysisAgent:
    def __init__(self, db_path, output_dir="output"):
        self.db_path = os.path.abspath(db_path)
        if not os.path.isfile(self.db_path):
            print(f"[!] Error: Database file not found at {self.db_path}")
            sys.exit(1)
            
        self.output_dir = output_dir
        self.report_filename = "Strategic_Business_Report.pdf"
        self.report_path = os.path.join(output_dir, self.report_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        self.tables = []
        self.total_records = 0
        self.email_insights = [] 
        self.generated_images = [] 
        
        self.kpi_map = {
            'Revenue': {'table_tag': 'sales_transaction', 'col_tag': 'net_sales_amount'},
            'Receivables (AR)': {'table_tag': 'ar_detail', 'col_tag': 'amount'}, 
            'Pipeline': {'table_tag': 'pipeline', 'col_tag': 'quote_amount'},
            'Orders': {'table_tag': 'sales_order', 'col_tag': 'order_amount'}
        }

    def run(self, sender, password, recipient):
        print(f"[*] Starting Analysis on {os.path.basename(self.db_path)}...")
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        
        try:
            self._discover_schema(conn)
            self._generate_report(conn)
            self._send_email(sender, password, recipient)
        finally:
            conn.close()

    def _discover_schema(self, conn):
        raw_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", conn)
        clean_names = set()
        for t in raw_tables['name']:
            clean_name = t.replace('"', '').replace("'", "")
            clean_names.add(clean_name)
        self.tables = sorted(list(clean_names))
        
        for t in self.tables:
            try:
                c = pd.read_sql_query(f"SELECT COUNT(*) as c FROM \"{t}\"", conn)['c'][0]
                self.total_records += c
            except: pass

    def _smart_date_parser(self, series):
        s_clean = series.astype(str).str.split('.').str[0]
        dt = pd.to_datetime(s_clean, errors='coerce')
        if dt.notna().mean() < 0.5:
            try:
                dt = pd.to_datetime(s_clean, format='%Y%m%d', errors='coerce')
            except: pass
        return dt

    def _get_max_date(self, conn, table):
        try:
            df = pd.read_sql_query(f"SELECT * FROM \"{table}\" LIMIT 1", conn)
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                q = f"SELECT MAX(\"{date_cols[0]}\") as d FROM \"{table}\""
                val = pd.read_sql_query(q, conn)['d'][0]
                return self._smart_date_parser(pd.Series([val])).iloc[0]
        except: pass
        return None

    def _generate_report(self, conn):
        with PdfPages(self.report_path) as pdf:
            self._page_executive_summary(conn, pdf)
            self._page_ar_aging(conn, pdf)
            self._page_data_quality(conn, pdf)
            self._page_deep_dives(conn, pdf)

    def _page_executive_summary(self, conn, pdf):
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.05, 0.95, "Strategic Business Report", fontsize=24, weight='bold', color='#2c3e50')
        plt.text(0.05, 0.91, f"Source: {os.path.basename(self.db_path)} | Generated: {datetime.now().strftime('%Y-%m-%d')}", fontsize=12, color='gray')
        plt.text(0.05, 0.80, "Executive KPIs (USD)", fontsize=16, weight='bold', color='#16a085')
        y = 0.75
        
        for kpi_name, tags in self.kpi_map.items():
            table_name = next((t for t in self.tables if tags['table_tag'] in t.lower()), None)
            if table_name:
                try:
                    cols = pd.read_sql_query(f"SELECT * FROM \"{table_name}\" LIMIT 1", conn).columns
                    valid_cols = [c for c in cols if tags['col_tag'] in c.lower() and 'tcy' not in c.lower() and 'gpb' not in c.lower()]
                    
                    if valid_cols:
                        col = valid_cols[0]
                        val = pd.read_sql_query(f"SELECT SUM(\"{col}\") as s FROM \"{table_name}\"", conn)['s'][0]
                        max_date = self._get_max_date(conn, table_name)
                        date_str = max_date.strftime('%Y-%m-%d') if max_date else "N/A"
                        
                        if val:
                            plt.text(0.05, y, f"{kpi_name}: ${val:,.2f}", fontsize=14, weight='bold')
                            plt.text(0.05, y-0.02, f"Source: {table_name}.{col} | As of: {date_str} | Curr: USD", fontsize=10, color='#555555')
                            y -= 0.07
                            # ADDED: Ensure this goes to email
                            self.email_insights.append(f"{kpi_name}: ${val:,.2f} (As of {date_str})")
                except: pass
        pdf.savefig()
        plt.close()

    def _page_ar_aging(self, conn, pdf):
        ar_table = next((t for t in self.tables if 'ar_detail' in t.lower()), None)
        if not ar_table: return

        try:
            df = pd.read_sql_query(f"SELECT * FROM \"{ar_table}\"", conn)
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            amt_cols = [c for c in df.columns if 'amount' in c.lower() and 'tcy' not in c.lower()]
            
            if date_cols and amt_cols:
                df['clean_date'] = self._smart_date_parser(df[date_cols[0]])
                df = df.dropna(subset=['clean_date'])
                if df.empty: return
                
                snapshot_date = df['clean_date'].max()
                df['days_overdue'] = (snapshot_date - df['clean_date']).dt.days
                
                bins = [0, 30, 60, 90, 9999]
                labels = ['0-30 Days', '31-60 Days', '61-90 Days', '90+ Days']
                df['bucket'] = pd.cut(df['days_overdue'], bins=bins, labels=labels)
                aging = df.groupby('bucket')[amt_cols[0]].sum()
                
                # Insight Calculation
                risk_pct = (aging.get('90+ Days', 0) / aging.sum()) * 100 if aging.sum() else 0
                # FIXED: Explicitly append to email list
                self.email_insights.append(f"AR Risk: {risk_pct:.1f}% of receivables are >90 days overdue.")

                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x=aging.index, y=aging.values, palette='OrRd')
                plt.title(f"AR Aging Profile (Snapshot: {snapshot_date.date()})", fontsize=14)
                plt.ylabel("Outstanding Amount (USD)")
                for i, v in enumerate(aging.values):
                    ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)

                png_path = os.path.join(self.output_dir, "chart2.png")
                plt.savefig(png_path, dpi=300)
                self.generated_images.append(png_path)
                pdf.savefig()
                plt.close()
        except: pass

    def _page_data_quality(self, conn, pdf):
        col_dq = []
        for t in self.tables:
            if any(x in t.lower() for x in ['customer', 'item', 'geo', 'org']):
                try:
                    df = pd.read_sql_query(f"SELECT * FROM \"{t}\" LIMIT 1000", conn)
                    for col in df.columns:
                        null_pct = df[col].isnull().mean() * 100
                        if null_pct > 10:
                            col_dq.append({'Table': t, 'Column': col, 'Missing %': null_pct})
                except: pass
        
        if col_dq:
            df_dq = pd.DataFrame(col_dq).sort_values('Missing %', ascending=False).head(8)
            
            # FIXED: Explicitly append to email list
            top_issue = df_dq.iloc[0]
            self.email_insights.append(f"Data Quality: Column '{top_issue['Column']}' is missing {top_issue['Missing %']:.1f}% data.")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_dq, x='Missing %', y='Column', hue='Table', dodge=False)
            plt.title("Top Critical Data Quality Gaps", fontsize=14)
            plt.xlabel("Percentage of Null Values")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
            plt.tight_layout()
            
            png_path = os.path.join(self.output_dir, "chart1.png")
            plt.savefig(png_path, dpi=300)
            self.generated_images.append(png_path)
            pdf.savefig()
            plt.close()

    def _page_deep_dives(self, conn, pdf):
        fact_table = next((t for t in self.tables if 'sales_transaction' in t.lower()), None)
        if fact_table:
            try:
                df = pd.read_sql_query(f"SELECT * FROM \"{fact_table}\" LIMIT 3000", conn)
                cols = [c for c in df.select_dtypes(include=np.number).columns 
                        if not any(x in c.lower() for x in ['id', 'key', 'code', 'line', 'type'])]
                
                if len(cols) > 2:
                    corr = df[cols].corr()
                    actions = []
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    for c in upper.columns:
                        for r in upper.index:
                            if upper.loc[r, c] > 0.95:
                                keep = c if 'tcy' in c.lower() else r 
                                drop = r if keep == c else c
                                actions.append(f"• DROP '{drop}' (Duplicate of '{keep}')")

                    plt.figure(figsize=(10, 8))
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={"shrink": .5})
                    plt.title(f"Metric Redundancy Matrix: {fact_table}", fontsize=14)
                    if actions:
                        plt.figtext(0.5, 0.01, "Optimization: " + " ".join(actions[:2]), ha="center", fontsize=8, bbox={"facecolor":"orange", "alpha":0.1})
                    
                    png_path = os.path.join(self.output_dir, "chart3.png")
                    plt.savefig(png_path, dpi=300)
                    self.generated_images.append(png_path)
                    pdf.savefig()
                    plt.close()
            except: pass

    def _send_email(self, sender, password, recipient):
        msg = MIMEMultipart()
        msg['Subject'] = f"Database Analysis Report - {os.path.basename(self.db_path)}"
        msg['From'] = sender
        msg['To'] = recipient
        
        # Fallback if no insights generated
        while len(self.email_insights) < 3: self.email_insights.append("Detailed analysis available in attached report.")

        # --- FORMATTED EMAIL BODY (As requested) ---
        body = f"""Dear Recipient,

Please find the automated database analysis report below.

=== DATABASE SUMMARY ===
- Total Tables: {len(self.tables)}
- Total Records: {self.total_records:,}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

=== KEY INSIGHTS ===
1. {self.email_insights[0]}
2. {self.email_insights[1]}
3. {self.email_insights[2]}

PDF Report & High-Res Charts Attached.

Best regards,
bob_the_builder
AI CODEFIX 2025
"""
        msg.attach(MIMEText(body, 'plain'))
        
        if os.path.exists(self.report_path):
            with open(self.report_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=self.report_filename)
                part['Content-Disposition'] = f'attachment; filename="{self.report_filename}"'
                msg.attach(part)
        
        for img_path in self.generated_images:
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img_part = MIMEImage(f.read())
                    img_part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(img_path))
                    msg.attach(img_part)
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
            print(f"[*] Email sent successfully to {recipient}")
        except Exception as e:
            print(f"[!] Email Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--sender", default="your_email@gmail.com")
    parser.add_argument("--password", default="your_app_password")
    args = parser.parse_args()
    
    agent = SalesAnalysisAgent(args.db, output_dir="output")
    agent.run(args.sender, args.password, args.email)