import os
import io
import json
import logging
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

# ✅ Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ✅ S3 Client
s3 = boto3.client("s3")

# ✅ Environment Variables
BUCKET = os.environ.get("BUCKET_NAME", "dataops-pipeline-bucket")
INPUT_KEY = os.environ.get("INPUT_KEY", "diabetes_preprocessed.csv")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "processed/")

# ✅ Helper functions
def _now_id():
    """Generate timestamp for versioning"""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _put_text(bucket, key, text):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))

def _put_json(bucket, key, obj):
    _put_text(bucket, key, json.dumps(obj, indent=2, default=str))

def lambda_handler(event, context):
    run_id = _now_id()
    run_log_key = f"{OUTPUT_PREFIX}run_{run_id}.json"

    try:
        logger.info(f"🚀 Starting pipeline for s3://{BUCKET}/{INPUT_KEY}")

        # 1. Read CSV from S3
        response = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        logger.info(f"📊 Dataset shape: {df.shape}")
        logger.info(f"🧼 Missing values before cleaning:\n{df.isnull().sum().to_string()}")

        # 2. Impute numeric columns with mean
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        # 3. Normalize numeric columns (Z-score)
        if num_cols:
            stds = df[num_cols].std(ddof=0).replace(0, np.nan)
            df[num_cols] = (df[num_cols] - df[num_cols].mean()) / stds
            df[num_cols] = df[num_cols].fillna(0.0)

        # 4. Correlation matrix
        corr_key = f"{OUTPUT_PREFIX}corr_{run_id}.csv"
        if num_cols:
            corr_df = df[num_cols].corr()
            _put_text(BUCKET, corr_key, corr_df.to_csv(index=True))
            logger.info(f"📈 Correlation matrix saved to s3://{BUCKET}/{corr_key}")

        # 5. Summary stats and dtypes
        summary_key = f"{OUTPUT_PREFIX}summary_{run_id}.json"
        dtypes_key = f"{OUTPUT_PREFIX}dtypes_{run_id}.json"

        summary = {
            "shape": {"rows": df.shape[0], "cols": df.shape[1]},
            "missing_after_clean": df.isnull().sum().to_dict(),
            "describe_numeric": df[num_cols].describe().to_dict() if num_cols else {}
        }
        dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

        _put_json(BUCKET, summary_key, summary)
        _put_json(BUCKET, dtypes_key, dtypes)

        # 6. Save cleaned CSV
        cleaned_key = f"{OUTPUT_PREFIX}cleaned_data_{run_id}.csv"
        out_buf = io.StringIO()
        df.to_csv(out_buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=cleaned_key, Body=out_buf.getvalue())
        logger.info(f"✅ Cleaned file saved to s3://{BUCKET}/{cleaned_key}")

        # 7. Save run log
        run_log = {
            "run_id": run_id,
            "bucket": BUCKET,
            "input_key": INPUT_KEY,
            "output_prefix": OUTPUT_PREFIX,
            "outputs": {
                "cleaned_csv": f"s3://{BUCKET}/{cleaned_key}",
                "corr_csv": f"s3://{BUCKET}/{corr_key}" if num_cols else None,
                "summary_json": f"s3://{BUCKET}/{summary_key}",
                "dtypes_json": f"s3://{BUCKET}/{dtypes_key}",
            },
            "status": "SUCCEEDED",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        _put_json(BUCKET, run_log_key, run_log)
        logger.info(f"📝 Run log saved to s3://{BUCKET}/{run_log_key}")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Pipeline executed successfully!", "run_id": run_id})
        }

    except ClientError as e:
        logger.error(f"AWS ClientError: {e}")
        _put_json(BUCKET, run_log_key, {"status": "FAILED", "error": str(e)})
        return {"statusCode": 500, "body": f"ClientError: {e}"}

    except Exception as e:
        logger.exception("Unhandled exception")
        _put_json(BUCKET, run_log_key, {"status": "FAILED", "error": str(e)})
        return {"statusCode": 500, "body": f"Error: {e}"}