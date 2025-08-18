"""

Simple and reliable CSV processor

"""

import pandas as pd
import base64
import io
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Simple CSV file processor"""

    def __init__(self):
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    def process_file(self, contents: str, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process uploaded CSV file

        Args:
            contents: Base64 encoded file content from dcc.Upload
            filename: Name of the uploaded file

        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        try:
            # Decode file content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Try different encodings
            df = None
            encoding_used = None
            for encoding in self.supported_encodings:
                try:
                    csv_string = decoded.decode(encoding)
                    df = pd.read_csv(io.StringIO(csv_string))
                    encoding_used = encoding
                    logger.info(f"Successfully decoded {filename} with {encoding}")
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                    continue

            if df is None:
                # Last resort: use utf-8 with error replacement
                csv_string = decoded.decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(csv_string))
                encoding_used = 'utf-8 (with errors replaced)'

            # Validate DataFrame
            if df.empty:
                raise ValueError("CSV file is empty")

            # Clean the DataFrame
            df = self._clean_dataframe(df)

            # Generate metadata
            metadata = self._generate_metadata(df, filename, encoding_used)

            logger.info(f"Successfully processed {filename}: {df.shape}")
            return df, metadata

        except Exception as e:
            logger.error(f"Error processing CSV {filename}: {e}")
            raise

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Replace infinite values with NaN
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df

    def _generate_metadata(self, df: pd.DataFrame, filename: str, encoding: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for the DataFrame"""
        return {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_type': 'csv',
            'encoding': encoding,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'has_missing_values': df.isnull().any().any(),
            'duplicate_rows': df.duplicated().sum()
        }
