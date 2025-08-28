"""
FraudLens Integration Manager
Handles connections to external systems and data sources
"""

import os
import json
import asyncio
import imaplib
import smtplib
import email
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import boto3
import psycopg2
from pymongo import MongoClient
import redis
import pika
from kafka import KafkaProducer, KafkaConsumer
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import requests
from loguru import logger


@dataclass
class EmailConfig:
    """Email configuration"""
    imap_host: str
    imap_port: int = 993
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_ssl: bool = True


@dataclass
class StorageConfig:
    """Cloud storage configuration"""
    provider: str  # "s3", "gcs", "azure"
    bucket: str
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    connection_string: Optional[str] = None


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str  # "postgresql", "mongodb", "mysql", "redis"
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False


@dataclass
class MessageQueueConfig:
    """Message queue configuration"""
    type: str  # "rabbitmq", "kafka", "sqs", "pubsub"
    host: Optional[str] = None
    port: Optional[int] = None
    topic: Optional[str] = None
    queue: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None


@dataclass
class SIEMConfig:
    """SIEM system configuration"""
    type: str  # "splunk", "elastic", "datadog", "sumologic"
    endpoint: str
    api_key: Optional[str] = None
    index: Optional[str] = None
    source_type: Optional[str] = None


@dataclass
class SyncResult:
    """Data synchronization result"""
    success: bool
    records_synced: int
    errors: List[str]
    duration_seconds: float
    metadata: Dict[str, Any]


class EmailConnection:
    """Email integration handler"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.imap = None
        self.smtp = None
    
    async def connect(self):
        """Connect to email servers"""
        
        # Connect to IMAP
        if self.config.use_ssl:
            self.imap = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port)
        else:
            self.imap = imaplib.IMAP4(self.config.imap_host, self.config.imap_port)
        
        self.imap.login(self.config.username, self.config.password)
        logger.info(f"Connected to IMAP server: {self.config.imap_host}")
        
        # Connect to SMTP if configured
        if self.config.smtp_host:
            self.smtp = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            if self.config.use_ssl:
                self.smtp.starttls()
            self.smtp.login(self.config.username, self.config.password)
            logger.info(f"Connected to SMTP server: {self.config.smtp_host}")
    
    async def fetch_emails(
        self,
        folder: str = "INBOX",
        criteria: str = "UNSEEN",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch emails from server"""
        
        emails = []
        
        try:
            self.imap.select(folder)
            
            # Search for emails
            _, message_ids = self.imap.search(None, criteria)
            
            for msg_id in message_ids[0].split()[:limit]:
                _, msg_data = self.imap.fetch(msg_id, "(RFC822)")
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Extract email data
                        email_data = {
                            "id": msg_id.decode(),
                            "from": msg["From"],
                            "to": msg["To"],
                            "subject": msg["Subject"],
                            "date": msg["Date"],
                            "body": self._extract_body(msg),
                            "attachments": self._extract_attachments(msg),
                            "headers": dict(msg.items())
                        }
                        
                        emails.append(email_data)
            
            logger.info(f"Fetched {len(emails)} emails from {folder}")
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
        
        return emails
    
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Send email"""
        
        if not self.smtp:
            logger.error("SMTP not configured")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.config.username
            msg["To"] = to
            msg["Subject"] = subject
            
            # Add text and HTML parts
            msg.attach(MIMEText(body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))
            
            self.smtp.send_message(msg)
            logger.info(f"Email sent to {to}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _extract_body(self, msg) -> str:
        """Extract email body"""
        
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        
        return body
    
    def _extract_attachments(self, msg) -> List[Dict[str, Any]]:
        """Extract email attachments"""
        
        attachments = []
        
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    content = part.get_payload(decode=True)
                    attachments.append({
                        "filename": filename,
                        "size": len(content),
                        "content_type": part.get_content_type(),
                        "content": content
                    })
        
        return attachments
    
    async def disconnect(self):
        """Disconnect from email servers"""
        
        if self.imap:
            self.imap.logout()
        if self.smtp:
            self.smtp.quit()


class StorageConnection:
    """Cloud storage integration handler"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.client = None
    
    async def connect(self):
        """Connect to cloud storage"""
        
        if self.config.provider == "s3":
            self.client = boto3.client(
                "s3",
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key
            )
            logger.info(f"Connected to S3 bucket: {self.config.bucket}")
            
        elif self.config.provider == "gcs":
            self.client = gcs.Client()
            logger.info(f"Connected to GCS bucket: {self.config.bucket}")
            
        elif self.config.provider == "azure":
            self.client = BlobServiceClient.from_connection_string(
                self.config.connection_string
            )
            logger.info(f"Connected to Azure blob storage: {self.config.bucket}")
    
    async def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload file to storage"""
        
        if not object_name:
            object_name = Path(file_path).name
        
        try:
            if self.config.provider == "s3":
                self.client.upload_file(
                    file_path,
                    self.config.bucket,
                    object_name,
                    ExtraArgs={"Metadata": metadata} if metadata else None
                )
                
            elif self.config.provider == "gcs":
                bucket = self.client.bucket(self.config.bucket)
                blob = bucket.blob(object_name)
                blob.upload_from_filename(file_path)
                if metadata:
                    blob.metadata = metadata
                    blob.patch()
                    
            elif self.config.provider == "azure":
                blob_client = self.client.get_blob_client(
                    container=self.config.bucket,
                    blob=object_name
                )
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, metadata=metadata)
            
            logger.info(f"Uploaded {file_path} to {self.config.provider}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    async def download_file(
        self,
        object_name: str,
        file_path: str
    ) -> bool:
        """Download file from storage"""
        
        try:
            if self.config.provider == "s3":
                self.client.download_file(
                    self.config.bucket,
                    object_name,
                    file_path
                )
                
            elif self.config.provider == "gcs":
                bucket = self.client.bucket(self.config.bucket)
                blob = bucket.blob(object_name)
                blob.download_to_filename(file_path)
                
            elif self.config.provider == "azure":
                blob_client = self.client.get_blob_client(
                    container=self.config.bucket,
                    blob=object_name
                )
                with open(file_path, "wb") as data:
                    data.write(blob_client.download_blob().readall())
            
            logger.info(f"Downloaded {object_name} from {self.config.provider}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """List objects in storage"""
        
        objects = []
        
        try:
            if self.config.provider == "s3":
                response = self.client.list_objects_v2(
                    Bucket=self.config.bucket,
                    Prefix=prefix or "",
                    MaxKeys=limit
                )
                
                for obj in response.get("Contents", []):
                    objects.append({
                        "name": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"],
                        "etag": obj["ETag"]
                    })
                    
            elif self.config.provider == "gcs":
                bucket = self.client.bucket(self.config.bucket)
                blobs = bucket.list_blobs(prefix=prefix, max_results=limit)
                
                for blob in blobs:
                    objects.append({
                        "name": blob.name,
                        "size": blob.size,
                        "modified": blob.updated,
                        "etag": blob.etag
                    })
                    
            elif self.config.provider == "azure":
                container_client = self.client.get_container_client(self.config.bucket)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                
                for blob in blobs:
                    objects.append({
                        "name": blob.name,
                        "size": blob.size,
                        "modified": blob.last_modified,
                        "etag": blob.etag
                    })
            
            logger.info(f"Listed {len(objects)} objects from {self.config.provider}")
            
        except Exception as e:
            logger.error(f"List objects failed: {e}")
        
        return objects
    
    async def delete_object(self, object_name: str) -> bool:
        """Delete object from storage"""
        
        try:
            if self.config.provider == "s3":
                self.client.delete_object(
                    Bucket=self.config.bucket,
                    Key=object_name
                )
                
            elif self.config.provider == "gcs":
                bucket = self.client.bucket(self.config.bucket)
                blob = bucket.blob(object_name)
                blob.delete()
                
            elif self.config.provider == "azure":
                blob_client = self.client.get_blob_client(
                    container=self.config.bucket,
                    blob=object_name
                )
                blob_client.delete_blob()
            
            logger.info(f"Deleted {object_name} from {self.config.provider}")
            return True
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


class DatabaseConnection:
    """Database integration handler"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.client = None
    
    async def connect(self):
        """Connect to database"""
        
        try:
            if self.config.type == "postgresql":
                self.connection = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    sslmode="require" if self.config.ssl else "disable"
                )
                logger.info(f"Connected to PostgreSQL: {self.config.database}")
                
            elif self.config.type == "mongodb":
                uri = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
                self.client = MongoClient(uri, ssl=self.config.ssl)
                self.connection = self.client[self.config.database]
                logger.info(f"Connected to MongoDB: {self.config.database}")
                
            elif self.config.type == "redis":
                self.connection = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    ssl=self.config.ssl,
                    decode_responses=True
                )
                logger.info(f"Connected to Redis: {self.config.host}")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute database query"""
        
        results = []
        
        try:
            if self.config.type == "postgresql":
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    for row in cursor.fetchall():
                        results.append(dict(zip(columns, row)))
                
                self.connection.commit()
                cursor.close()
                
            elif self.config.type == "mongodb":
                # Parse MongoDB-style query
                collection_name, operation = query.split(".", 1)
                collection = self.connection[collection_name]
                
                if operation.startswith("find"):
                    cursor = collection.find(params or {})
                    results = list(cursor)
                    
            elif self.config.type == "redis":
                # Execute Redis command
                command_parts = query.split()
                command = command_parts[0].lower()
                args = command_parts[1:]
                
                result = getattr(self.connection, command)(*args)
                results = [{"result": result}]
            
            logger.info(f"Query executed, returned {len(results)} results")
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        
        return results
    
    async def insert_data(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> int:
        """Insert data into database"""
        
        if isinstance(data, dict):
            data = [data]
        
        inserted = 0
        
        try:
            if self.config.type == "postgresql":
                cursor = self.connection.cursor()
                
                for record in data:
                    columns = ", ".join(record.keys())
                    placeholders = ", ".join(["%s"] * len(record))
                    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                    
                    cursor.execute(query, tuple(record.values()))
                    inserted += cursor.rowcount
                
                self.connection.commit()
                cursor.close()
                
            elif self.config.type == "mongodb":
                collection = self.connection[table]
                result = collection.insert_many(data)
                inserted = len(result.inserted_ids)
                
            elif self.config.type == "redis":
                for record in data:
                    key = f"{table}:{record.get('id', hashlib.md5(json.dumps(record).encode()).hexdigest())}"
                    self.connection.hset(key, mapping=record)
                    inserted += 1
            
            logger.info(f"Inserted {inserted} records into {table}")
            
        except Exception as e:
            logger.error(f"Data insertion failed: {e}")
            raise
        
        return inserted
    
    async def disconnect(self):
        """Disconnect from database"""
        
        if self.connection:
            if self.config.type == "postgresql":
                self.connection.close()
            elif self.config.type == "mongodb":
                self.client.close()
            elif self.config.type == "redis":
                self.connection.close()
            
            logger.info(f"Disconnected from {self.config.type}")


class MessageQueueConnection:
    """Message queue integration handler"""
    
    def __init__(self, config: MessageQueueConfig):
        self.config = config
        self.connection = None
        self.channel = None
        self.producer = None
        self.consumer = None
    
    async def connect(self):
        """Connect to message queue"""
        
        try:
            if self.config.type == "rabbitmq":
                credentials = pika.PlainCredentials(
                    self.config.credentials.get("username"),
                    self.config.credentials.get("password")
                )
                parameters = pika.ConnectionParameters(
                    host=self.config.host,
                    port=self.config.port,
                    credentials=credentials
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                if self.config.queue:
                    self.channel.queue_declare(queue=self.config.queue, durable=True)
                
                logger.info(f"Connected to RabbitMQ: {self.config.host}")
                
            elif self.config.type == "kafka":
                self.producer = KafkaProducer(
                    bootstrap_servers=f"{self.config.host}:{self.config.port}",
                    value_serializer=lambda v: json.dumps(v).encode()
                )
                self.consumer = KafkaConsumer(
                    self.config.topic,
                    bootstrap_servers=f"{self.config.host}:{self.config.port}",
                    value_deserializer=lambda m: json.loads(m.decode())
                )
                logger.info(f"Connected to Kafka: {self.config.host}")
                
        except Exception as e:
            logger.error(f"Message queue connection failed: {e}")
            raise
    
    async def send_message(
        self,
        message: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> bool:
        """Send message to queue"""
        
        try:
            if self.config.type == "rabbitmq":
                self.channel.basic_publish(
                    exchange="",
                    routing_key=routing_key or self.config.queue,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2  # Make message persistent
                    )
                )
                logger.info(f"Message sent to RabbitMQ queue: {self.config.queue}")
                
            elif self.config.type == "kafka":
                future = self.producer.send(
                    self.config.topic,
                    value=message
                )
                self.producer.flush()
                logger.info(f"Message sent to Kafka topic: {self.config.topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(
        self,
        callback,
        max_messages: Optional[int] = None
    ):
        """Receive messages from queue"""
        
        messages_processed = 0
        
        try:
            if self.config.type == "rabbitmq":
                def rabbitmq_callback(ch, method, properties, body):
                    nonlocal messages_processed
                    message = json.loads(body)
                    callback(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                    messages_processed += 1
                    if max_messages and messages_processed >= max_messages:
                        ch.stop_consuming()
                
                self.channel.basic_consume(
                    queue=self.config.queue,
                    on_message_callback=rabbitmq_callback,
                    auto_ack=False
                )
                self.channel.start_consuming()
                
            elif self.config.type == "kafka":
                for message in self.consumer:
                    callback(message.value)
                    
                    messages_processed += 1
                    if max_messages and messages_processed >= max_messages:
                        break
            
            logger.info(f"Processed {messages_processed} messages")
            
        except Exception as e:
            logger.error(f"Failed to receive messages: {e}")
    
    async def disconnect(self):
        """Disconnect from message queue"""
        
        if self.config.type == "rabbitmq" and self.connection:
            self.connection.close()
        elif self.config.type == "kafka":
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
        
        logger.info(f"Disconnected from {self.config.type}")


class SIEMConnection:
    """SIEM system integration handler"""
    
    def __init__(self, config: SIEMConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers["Authorization"] = f"Bearer {config.api_key}"
    
    async def send_event(
        self,
        event_data: Dict[str, Any],
        severity: str = "info"
    ) -> bool:
        """Send event to SIEM"""
        
        try:
            if self.config.type == "splunk":
                url = f"{self.config.endpoint}/services/collector/event"
                payload = {
                    "event": event_data,
                    "source": "fraudlens",
                    "sourcetype": self.config.source_type or "json",
                    "index": self.config.index or "main",
                    "time": datetime.utcnow().timestamp()
                }
                
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                
            elif self.config.type == "elastic":
                url = f"{self.config.endpoint}/{self.config.index}/_doc"
                payload = {
                    **event_data,
                    "@timestamp": datetime.utcnow().isoformat(),
                    "severity": severity,
                    "source": "fraudlens"
                }
                
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                
            elif self.config.type == "datadog":
                url = f"{self.config.endpoint}/v1/logs"
                payload = {
                    "ddsource": "fraudlens",
                    "ddtags": f"env:production,severity:{severity}",
                    "hostname": "fraudlens-api",
                    "message": json.dumps(event_data)
                }
                
                response = self.session.post(url, json=payload)
                response.raise_for_status()
            
            logger.info(f"Event sent to {self.config.type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event to SIEM: {e}")
            return False
    
    async def query_events(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query events from SIEM"""
        
        events = []
        
        try:
            if self.config.type == "splunk":
                url = f"{self.config.endpoint}/services/search/jobs"
                
                # Create search job
                search_query = f"search {query} earliest={start_time.isoformat()} latest={end_time.isoformat()} | head {limit}"
                response = self.session.post(
                    url,
                    data={"search": search_query, "output_mode": "json"}
                )
                job_id = response.json()["sid"]
                
                # Get results
                results_url = f"{url}/{job_id}/results"
                response = self.session.get(results_url, params={"output_mode": "json"})
                events = response.json().get("results", [])
                
            elif self.config.type == "elastic":
                url = f"{self.config.endpoint}/{self.config.index}/_search"
                query_body = {
                    "query": {
                        "bool": {
                            "must": [
                                {"query_string": {"query": query}},
                                {
                                    "range": {
                                        "@timestamp": {
                                            "gte": start_time.isoformat(),
                                            "lte": end_time.isoformat()
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "size": limit
                }
                
                response = self.session.post(url, json=query_body)
                hits = response.json().get("hits", {}).get("hits", [])
                events = [hit["_source"] for hit in hits]
            
            logger.info(f"Retrieved {len(events)} events from {self.config.type}")
            
        except Exception as e:
            logger.error(f"Failed to query SIEM: {e}")
        
        return events


class IntegrationManager:
    """
    Manages all external integrations
    """
    
    def __init__(self):
        self.connections = {}
        self.health_status = {}
    
    async def connect_email(self, config: EmailConfig) -> EmailConnection:
        """Connect to email server"""
        
        connection = EmailConnection(config)
        await connection.connect()
        
        connection_id = f"email_{config.imap_host}"
        self.connections[connection_id] = connection
        self.health_status[connection_id] = True
        
        return connection
    
    async def connect_storage(self, config: StorageConfig) -> StorageConnection:
        """Connect to cloud storage"""
        
        connection = StorageConnection(config)
        await connection.connect()
        
        connection_id = f"storage_{config.provider}_{config.bucket}"
        self.connections[connection_id] = connection
        self.health_status[connection_id] = True
        
        return connection
    
    async def connect_database(self, config: DatabaseConfig) -> DatabaseConnection:
        """Connect to database"""
        
        connection = DatabaseConnection(config)
        await connection.connect()
        
        connection_id = f"db_{config.type}_{config.database}"
        self.connections[connection_id] = connection
        self.health_status[connection_id] = True
        
        return connection
    
    async def connect_message_queue(self, config: MessageQueueConfig) -> MessageQueueConnection:
        """Connect to message queue"""
        
        connection = MessageQueueConnection(config)
        await connection.connect()
        
        connection_id = f"mq_{config.type}_{config.host}"
        self.connections[connection_id] = connection
        self.health_status[connection_id] = True
        
        return connection
    
    async def connect_siem(self, config: SIEMConfig) -> SIEMConnection:
        """Connect to SIEM system"""
        
        connection = SIEMConnection(config)
        
        connection_id = f"siem_{config.type}"
        self.connections[connection_id] = connection
        self.health_status[connection_id] = True
        
        return connection
    
    async def sync_data(
        self,
        source: str,
        destination: str,
        transform_func: Optional[callable] = None,
        batch_size: int = 100
    ) -> SyncResult:
        """Sync data between systems"""
        
        start_time = datetime.now()
        records_synced = 0
        errors = []
        
        try:
            source_conn = self.connections.get(source)
            dest_conn = self.connections.get(destination)
            
            if not source_conn or not dest_conn:
                raise ValueError("Invalid source or destination")
            
            # Fetch data from source
            if isinstance(source_conn, DatabaseConnection):
                data = await source_conn.execute_query("SELECT * FROM fraud_events")
            elif isinstance(source_conn, StorageConnection):
                objects = await source_conn.list_objects()
                data = objects
            else:
                data = []
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                # Transform data if function provided
                if transform_func:
                    batch = [transform_func(record) for record in batch]
                
                # Write to destination
                if isinstance(dest_conn, DatabaseConnection):
                    inserted = await dest_conn.insert_data("fraud_events", batch)
                    records_synced += inserted
                elif isinstance(dest_conn, StorageConnection):
                    for record in batch:
                        success = await dest_conn.upload_file(
                            record.get("file_path"),
                            record.get("object_name")
                        )
                        if success:
                            records_synced += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return SyncResult(
                success=True,
                records_synced=records_synced,
                errors=errors,
                duration_seconds=duration,
                metadata={
                    "source": source,
                    "destination": destination,
                    "batch_size": batch_size
                }
            )
            
        except Exception as e:
            logger.error(f"Data sync failed: {e}")
            errors.append(str(e))
            
            return SyncResult(
                success=False,
                records_synced=records_synced,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={}
            )
    
    def is_healthy(self) -> bool:
        """Check if all connections are healthy"""
        return all(self.health_status.values()) if self.health_status else True
    
    async def disconnect_all(self):
        """Disconnect all connections"""
        
        for connection_id, connection in self.connections.items():
            try:
                if hasattr(connection, "disconnect"):
                    await connection.disconnect()
                logger.info(f"Disconnected: {connection_id}")
            except Exception as e:
                logger.error(f"Failed to disconnect {connection_id}: {e}")
        
        self.connections.clear()
        self.health_status.clear()