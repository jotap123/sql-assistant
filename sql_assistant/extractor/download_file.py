import os
import tempfile
from typing import Union
from io import BytesIO
from flask import Flask, send_file, Response

class FlaskDownloadHandler:
    def __init__(self, app: Flask):
        """
        Initialize the download handler with Flask routes
        
        Args:
            app (Flask): Flask application instance
        """
        self.app = app
        self.register_routes()
    
    def register_routes(self):
        """Register the download routes with the Flask app"""
        
        @self.app.route('/download/<format_type>')
        def download_result(format_type: str) -> Union[Response, tuple]:
            df = self.app.query_results  # Assuming you store the DataFrame in app context
            
            try:
                if format_type == 'csv':
                    buffer = BytesIO()
                    df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    
                    return send_file(
                        buffer,
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name='query_results.csv'
                    )
                
                elif format_type == 'parquet':
                    buffer = BytesIO()
                    df.to_parquet(buffer, index=False)
                    buffer.seek(0)
                    
                    return send_file(
                        buffer,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name='query_results.parquet'
                    )
                
                else:
                    return {"error": "Invalid format type"}, 400
                
            except Exception as e:
                return {"error": str(e)}, 500