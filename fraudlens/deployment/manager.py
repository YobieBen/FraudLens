"""
FraudLens Deployment Manager
Handles packaging, containerization, and cloud deployment
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import tempfile
import hashlib


@dataclass
class AppBundle:
    """Mac application bundle"""

    name: str
    version: str
    bundle_id: str
    executable_path: Path
    resources_path: Path
    info_plist: Dict[str, Any]
    size_mb: float
    signature: Optional[str] = None


@dataclass
class DockerImage:
    """Docker image information"""

    repository: str
    tag: str
    image_id: str
    size_mb: float
    layers: int
    created: datetime
    platforms: List[str]


@dataclass
class DeploymentStatus:
    """Kubernetes deployment status"""

    namespace: str
    name: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    conditions: List[Dict[str, Any]]
    endpoints: List[str]


@dataclass
class GatewayConfig:
    """API Gateway configuration"""

    provider: str  # "kong", "nginx", "aws", "cloudflare"
    endpoints: Dict[str, str]
    rate_limits: Dict[str, int]
    auth_enabled: bool
    ssl_enabled: bool
    cors_origins: List[str]


@dataclass
class MonitoringDashboard:
    """Monitoring dashboard configuration"""

    provider: str  # "grafana", "datadog", "newrelic"
    url: str
    dashboards: List[str]
    alerts: List[Dict[str, Any]]
    metrics_endpoint: str


class DeploymentManager:
    """
    Manages FraudLens deployment across platforms
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.deployment_dir = self.project_root / "deployment"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

    def package_for_mac(
        self,
        app_name: str = "FraudLens",
        version: str = "1.0.0",
        bundle_id: str = "com.fraudlens.app",
        sign_identity: Optional[str] = None,
    ) -> AppBundle:
        """
        Package FraudLens as a standalone Mac application
        """
        print("📦 Packaging FraudLens for macOS...")

        # Create app bundle structure
        app_dir = self.deployment_dir / f"{app_name}.app"
        contents_dir = app_dir / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"

        # Create directories
        for dir_path in [app_dir, contents_dir, macos_dir, resources_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create PyInstaller spec file
        spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{self.project_root}/fraudlens/__main__.py'],
    pathex=['{self.project_root}'],
    binaries=[],
    datas=[
        ('{self.project_root}/fraudlens', 'fraudlens'),
        ('{self.project_root}/configs', 'configs'),
        ('{self.project_root}/models', 'models'),
    ],
    hiddenimports=[
        'fraudlens',
        'torch',
        'torchvision',
        'transformers',
        'opencv-python',
        'pillow',
        'gradio',
        'fastapi',
        'uvicorn',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['test', 'tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch='arm64',  # For M4 Mac
    codesign_identity='{sign_identity or ""}',
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='{app_name}.app',
    icon='{resources_dir}/icon.icns',
    bundle_identifier='{bundle_id}',
    version='{version}',
    info_plist={{
        'NSHighResolutionCapable': True,
        'CFBundleDisplayName': '{app_name}',
        'CFBundleName': '{app_name}',
        'CFBundlePackageType': 'APPL',
        'CFBundleShortVersionString': '{version}',
        'CFBundleVersion': '{version}',
        'LSMinimumSystemVersion': '13.0',
        'NSRequiresAquaSystemAppearance': False,
        'NSCameraUsageDescription': 'FraudLens needs camera access for image fraud detection',
        'NSMicrophoneUsageDescription': 'FraudLens needs microphone access for audio fraud detection',
    }},
)
"""

        spec_path = self.deployment_dir / f"{app_name}.spec"
        with open(spec_path, "w") as f:
            f.write(spec_content)

        # Create Info.plist
        info_plist = {
            "CFBundleDisplayName": app_name,
            "CFBundleName": app_name,
            "CFBundleIdentifier": bundle_id,
            "CFBundleVersion": version,
            "CFBundleShortVersionString": version,
            "CFBundlePackageType": "APPL",
            "CFBundleSignature": "????",
            "CFBundleExecutable": app_name,
            "CFBundleIconFile": "icon.icns",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
            "NSRequiresAquaSystemAppearance": False,
            "LSApplicationCategoryType": "public.app-category.developer-tools",
        }

        plist_path = contents_dir / "Info.plist"
        subprocess.run(
            ["plutil", "-convert", "xml1", "-o", str(plist_path), "-"],
            input=json.dumps(info_plist),
            text=True,
            check=True,
        )

        # Build with PyInstaller
        print("🔨 Building with PyInstaller...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                str(spec_path),
                "--distpath",
                str(self.deployment_dir),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            raise RuntimeError("PyInstaller build failed")

        # Sign the app if identity provided
        if sign_identity:
            print(f"✍️ Signing app with identity: {sign_identity}")
            subprocess.run(
                ["codesign", "--deep", "--force", "--sign", sign_identity, str(app_dir)], check=True
            )

        # Calculate app size
        app_size = sum(f.stat().st_size for f in app_dir.rglob("*") if f.is_file()) / (1024**2)

        print(f"✅ Mac app bundle created: {app_dir}")
        print(f"   Size: {app_size:.2f} MB")

        return AppBundle(
            name=app_name,
            version=version,
            bundle_id=bundle_id,
            executable_path=macos_dir / app_name,
            resources_path=resources_dir,
            info_plist=info_plist,
            size_mb=app_size,
            signature=sign_identity,
        )

    def build_docker_image(
        self,
        repository: str = "fraudlens",
        tag: str = "latest",
        platforms: List[str] = ["linux/amd64", "linux/arm64"],
        push: bool = False,
        registry: Optional[str] = None,
    ) -> DockerImage:
        """
        Build multi-architecture Docker image
        """
        print(f"🐳 Building Docker image: {repository}:{tag}")

        # Ensure Dockerfile exists
        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            raise FileNotFoundError("Dockerfile not found")

        # Setup buildx for multi-platform builds
        print("Setting up Docker buildx...")
        subprocess.run(["docker", "buildx", "create", "--use"], check=True)

        # Build command
        build_cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            ",".join(platforms),
            "-t",
            f"{repository}:{tag}",
            "-f",
            str(dockerfile_path),
            ".",
        ]

        if push and registry:
            full_tag = f"{registry}/{repository}:{tag}"
            build_cmd.extend(["-t", full_tag, "--push"])
        else:
            build_cmd.append("--load")

        # Build image
        print(f"Building for platforms: {', '.join(platforms)}")
        result = subprocess.run(build_cmd, cwd=self.project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            raise RuntimeError("Docker build failed")

        # Get image info
        image_info = subprocess.run(
            ["docker", "images", repository, "--format", "json"], capture_output=True, text=True
        ).stdout

        if image_info:
            info = json.loads(image_info.split("\n")[0])
            image_id = info.get("ID", "unknown")
            size_mb = float(info.get("Size", "0").replace("MB", ""))
        else:
            image_id = "unknown"
            size_mb = 0.0

        print(f"✅ Docker image built: {repository}:{tag}")
        print(f"   Image ID: {image_id}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Platforms: {', '.join(platforms)}")

        return DockerImage(
            repository=repository,
            tag=tag,
            image_id=image_id,
            size_mb=size_mb,
            layers=0,  # TODO: Get actual layer count
            created=datetime.now(),
            platforms=platforms,
        )

    def deploy_to_kubernetes(
        self,
        namespace: str = "fraudlens",
        replicas: int = 3,
        image: str = "fraudlens:latest",
        cpu_request: str = "500m",
        memory_request: str = "1Gi",
        cpu_limit: str = "2000m",
        memory_limit: str = "4Gi",
    ) -> DeploymentStatus:
        """
        Deploy to Kubernetes cluster
        """
        print(f"☸️ Deploying to Kubernetes namespace: {namespace}")

        # Create namespace if it doesn't exist
        subprocess.run(["kubectl", "create", "namespace", namespace], capture_output=True)

        # Generate Kubernetes manifests
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraudlens
  namespace: {namespace}
  labels:
    app: fraudlens
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: fraudlens
  template:
    metadata:
      labels:
        app: fraudlens
    spec:
      containers:
      - name: fraudlens
        image: {image}
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 7860
          name: gradio
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
        env:
        - name: FRAUDLENS_ENV
          value: "production"
        - name: WORKERS
          value: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fraudlens-service
  namespace: {namespace}
spec:
  selector:
    app: fraudlens
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: gradio
    port: 7860
    targetPort: 7860
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraudlens-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraudlens
  minReplicas: {replicas}
  maxReplicas: {replicas * 3}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

        # Save and apply manifests
        manifest_path = self.deployment_dir / "k8s-deployment.yaml"
        with open(manifest_path, "w") as f:
            f.write(deployment_yaml)

        # Apply to cluster
        print("Applying Kubernetes manifests...")
        result = subprocess.run(
            ["kubectl", "apply", "-f", str(manifest_path)], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"❌ Deployment failed: {result.stderr}")
            raise RuntimeError("Kubernetes deployment failed")

        # Wait for deployment to be ready
        print("Waiting for deployment to be ready...")
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--for=condition=available",
                f"--namespace={namespace}",
                "deployment/fraudlens",
                "--timeout=300s",
            ],
            check=True,
        )

        # Get deployment status
        status_json = subprocess.run(
            ["kubectl", "get", "deployment", "fraudlens", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
        ).stdout

        status = json.loads(status_json)

        # Get service endpoints
        service_json = subprocess.run(
            ["kubectl", "get", "service", "fraudlens-service", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
        ).stdout

        service = json.loads(service_json)
        endpoints = []
        if "status" in service and "loadBalancer" in service["status"]:
            for ingress in service["status"]["loadBalancer"].get("ingress", []):
                ip = ingress.get("ip") or ingress.get("hostname")
                if ip:
                    endpoints.append(f"http://{ip}:8000")
                    endpoints.append(f"http://{ip}:7860")

        print(f"✅ Deployment successful")
        print(f"   Replicas: {status['status'].get('readyReplicas', 0)}/{replicas}")
        if endpoints:
            print(f"   Endpoints: {', '.join(endpoints)}")

        return DeploymentStatus(
            namespace=namespace,
            name="fraudlens",
            replicas=replicas,
            ready_replicas=status["status"].get("readyReplicas", 0),
            available_replicas=status["status"].get("availableReplicas", 0),
            conditions=status["status"].get("conditions", []),
            endpoints=endpoints,
        )

    def configure_api_gateway(
        self,
        provider: str = "nginx",
        domain: str = "api.fraudlens.com",
        ssl_cert: Optional[str] = None,
        ssl_key: Optional[str] = None,
        rate_limit: int = 1000,  # requests per minute
        enable_auth: bool = True,
    ) -> GatewayConfig:
        """
        Configure API Gateway
        """
        print(f"🌐 Configuring API Gateway with {provider}")

        if provider == "nginx":
            config = self._configure_nginx_gateway(
                domain, ssl_cert, ssl_key, rate_limit, enable_auth
            )
        elif provider == "kong":
            config = self._configure_kong_gateway(
                domain, ssl_cert, ssl_key, rate_limit, enable_auth
            )
        elif provider == "aws":
            config = self._configure_aws_api_gateway(domain, rate_limit, enable_auth)
        else:
            raise ValueError(f"Unsupported gateway provider: {provider}")

        print(f"✅ API Gateway configured")
        print(f"   Provider: {provider}")
        print(f"   Domain: {domain}")
        print(f"   Rate limit: {rate_limit} req/min")
        print(f"   Auth: {'Enabled' if enable_auth else 'Disabled'}")
        print(f"   SSL: {'Enabled' if ssl_cert else 'Disabled'}")

        return config

    def _configure_nginx_gateway(
        self,
        domain: str,
        ssl_cert: Optional[str],
        ssl_key: Optional[str],
        rate_limit: int,
        enable_auth: bool,
    ) -> GatewayConfig:
        """Configure NGINX as API Gateway"""

        nginx_config = f"""
upstream fraudlens_backend {{
    least_conn;
    server fraudlens-1:8000 max_fails=3 fail_timeout=30s;
    server fraudlens-2:8000 max_fails=3 fail_timeout=30s;
    server fraudlens-3:8000 max_fails=3 fail_timeout=30s;
}}

limit_req_zone $binary_remote_addr zone=api_limit:10m rate={rate_limit}r/m;

server {{
    listen 80;
    listen 443 ssl http2;
    server_name {domain};
    
    {'ssl_certificate ' + ssl_cert + ';' if ssl_cert else ''}
    {'ssl_certificate_key ' + ssl_key + ';' if ssl_key else ''}
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API versioning
    location /api/v1/ {{
        limit_req zone=api_limit burst=20 nodelay;
        
        {'auth_basic "FraudLens API";' if enable_auth else ''}
        {'auth_basic_user_file /etc/nginx/.htpasswd;' if enable_auth else ''}
        
        proxy_pass http://fraudlens_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }}
    
    # Health check endpoint
    location /health {{
        access_log off;
        proxy_pass http://fraudlens_backend/health;
    }}
    
    # Metrics endpoint (internal only)
    location /metrics {{
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://fraudlens_backend/metrics;
    }}
}}
"""

        config_path = self.deployment_dir / "nginx.conf"
        with open(config_path, "w") as f:
            f.write(nginx_config)

        return GatewayConfig(
            provider="nginx",
            endpoints={
                "api": f"https://{domain}/api/v1",
                "health": f"https://{domain}/health",
                "metrics": f"https://{domain}/metrics",
            },
            rate_limits={"default": rate_limit},
            auth_enabled=enable_auth,
            ssl_enabled=bool(ssl_cert),
            cors_origins=["*"],
        )

    def _configure_kong_gateway(
        self,
        domain: str,
        ssl_cert: Optional[str],
        ssl_key: Optional[str],
        rate_limit: int,
        enable_auth: bool,
    ) -> GatewayConfig:
        """Configure Kong API Gateway"""

        kong_config = {
            "services": [
                {
                    "name": "fraudlens-api",
                    "url": "http://fraudlens-backend:8000",
                    "routes": [
                        {
                            "name": "fraudlens-route",
                            "hosts": [domain],
                            "paths": ["/api/v1"],
                            "strip_path": True,
                        }
                    ],
                    "plugins": [
                        {
                            "name": "rate-limiting",
                            "config": {"minute": rate_limit, "policy": "cluster"},
                        },
                        {
                            "name": "cors",
                            "config": {
                                "origins": ["*"],
                                "methods": ["GET", "POST", "PUT", "DELETE"],
                                "headers": ["Content-Type", "Authorization"],
                            },
                        },
                    ],
                }
            ]
        }

        if enable_auth:
            kong_config["services"][0]["plugins"].append(
                {"name": "key-auth", "config": {"key_names": ["X-API-Key", "apikey"]}}
            )

        config_path = self.deployment_dir / "kong-config.json"
        with open(config_path, "w") as f:
            json.dump(kong_config, f, indent=2)

        return GatewayConfig(
            provider="kong",
            endpoints={"api": f"https://{domain}/api/v1", "admin": f"https://{domain}:8001"},
            rate_limits={"default": rate_limit},
            auth_enabled=enable_auth,
            ssl_enabled=bool(ssl_cert),
            cors_origins=["*"],
        )

    def _configure_aws_api_gateway(
        self, domain: str, rate_limit: int, enable_auth: bool
    ) -> GatewayConfig:
        """Configure AWS API Gateway"""

        # CloudFormation template for AWS API Gateway
        cf_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "FraudLensAPI": {
                    "Type": "AWS::ApiGateway::RestApi",
                    "Properties": {
                        "Name": "FraudLensAPI",
                        "EndpointConfiguration": {"Types": ["EDGE"]},
                    },
                },
                "UsagePlan": {
                    "Type": "AWS::ApiGateway::UsagePlan",
                    "Properties": {
                        "UsagePlanName": "FraudLensUsagePlan",
                        "Throttle": {"RateLimit": rate_limit, "BurstLimit": rate_limit * 2},
                    },
                },
            },
        }

        if enable_auth:
            cf_template["Resources"]["Authorizer"] = {
                "Type": "AWS::ApiGateway::Authorizer",
                "Properties": {
                    "Type": "COGNITO_USER_POOLS",
                    "Name": "FraudLensAuthorizer",
                    "RestApiId": {"Ref": "FraudLensAPI"},
                },
            }

        config_path = self.deployment_dir / "aws-api-gateway.json"
        with open(config_path, "w") as f:
            json.dump(cf_template, f, indent=2)

        return GatewayConfig(
            provider="aws",
            endpoints={
                "api": f"https://{domain}/api/v1",
                "console": "https://console.aws.amazon.com/apigateway",
            },
            rate_limits={"default": rate_limit},
            auth_enabled=enable_auth,
            ssl_enabled=True,  # AWS handles SSL
            cors_origins=["*"],
        )

    def setup_monitoring(
        self, provider: str = "grafana", metrics_port: int = 9090, alerting: bool = True
    ) -> MonitoringDashboard:
        """
        Setup monitoring dashboard
        """
        print(f"📊 Setting up monitoring with {provider}")

        if provider == "grafana":
            dashboard = self._setup_grafana_monitoring(metrics_port, alerting)
        elif provider == "datadog":
            dashboard = self._setup_datadog_monitoring(alerting)
        elif provider == "prometheus":
            dashboard = self._setup_prometheus_monitoring(metrics_port, alerting)
        else:
            raise ValueError(f"Unsupported monitoring provider: {provider}")

        print(f"✅ Monitoring dashboard configured")
        print(f"   Provider: {provider}")
        print(f"   URL: {dashboard.url}")
        print(f"   Dashboards: {', '.join(dashboard.dashboards)}")
        print(f"   Alerts: {len(dashboard.alerts)} configured")

        return dashboard

    def _setup_grafana_monitoring(self, metrics_port: int, alerting: bool) -> MonitoringDashboard:
        """Setup Grafana monitoring"""

        # Grafana dashboard JSON
        dashboard_config = {
            "dashboard": {
                "title": "FraudLens Monitoring",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{"expr": "rate(fraudlens_requests_total[5m])"}],
                    },
                    {
                        "title": "Latency",
                        "type": "graph",
                        "targets": [
                            {"expr": "histogram_quantile(0.95, fraudlens_request_duration_seconds)"}
                        ],
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [{"expr": "rate(fraudlens_errors_total[5m])"}],
                    },
                    {
                        "title": "Model Performance",
                        "type": "graph",
                        "targets": [{"expr": "fraudlens_model_accuracy"}],
                    },
                ],
            }
        }

        # Alerts configuration
        alerts = []
        if alerting:
            alerts = [
                {
                    "name": "HighErrorRate",
                    "condition": "rate(fraudlens_errors_total[5m]) > 0.05",
                    "severity": "critical",
                },
                {
                    "name": "HighLatency",
                    "condition": "histogram_quantile(0.95, fraudlens_request_duration_seconds) > 1",
                    "severity": "warning",
                },
                {
                    "name": "LowAccuracy",
                    "condition": "fraudlens_model_accuracy < 0.9",
                    "severity": "warning",
                },
            ]

        config_path = self.deployment_dir / "grafana-dashboard.json"
        with open(config_path, "w") as f:
            json.dump(dashboard_config, f, indent=2)

        return MonitoringDashboard(
            provider="grafana",
            url="http://localhost:3000",
            dashboards=["FraudLens Overview", "Model Performance", "System Metrics"],
            alerts=alerts,
            metrics_endpoint=f"http://localhost:{metrics_port}/metrics",
        )

    def _setup_datadog_monitoring(self, alerting: bool) -> MonitoringDashboard:
        """Setup Datadog monitoring"""

        # Datadog configuration
        datadog_config = {
            "api_key": "${DATADOG_API_KEY}",
            "site": "datadoghq.com",
            "logs_enabled": True,
            "apm_enabled": True,
            "process_config": {"enabled": True},
            "tags": ["service:fraudlens", "env:production"],
        }

        config_path = self.deployment_dir / "datadog.yaml"
        with open(config_path, "w") as f:
            yaml.dump(datadog_config, f)

        alerts = []
        if alerting:
            alerts = [
                {
                    "name": "fraudlens.high_error_rate",
                    "type": "metric",
                    "query": "avg(last_5m):avg:fraudlens.errors{*} > 0.05",
                }
            ]

        return MonitoringDashboard(
            provider="datadog",
            url="https://app.datadoghq.com",
            dashboards=["FraudLens APM", "Infrastructure", "Logs"],
            alerts=alerts,
            metrics_endpoint="datadog-agent:8126",
        )

    def _setup_prometheus_monitoring(
        self, metrics_port: int, alerting: bool
    ) -> MonitoringDashboard:
        """Setup Prometheus monitoring"""

        # Prometheus configuration
        prometheus_config = {
            "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
            "scrape_configs": [
                {
                    "job_name": "fraudlens",
                    "static_configs": [{"targets": [f"localhost:{metrics_port}"]}],
                }
            ],
        }

        if alerting:
            prometheus_config["rule_files"] = ["alerts.yml"]

        config_path = self.deployment_dir / "prometheus.yml"
        with open(config_path, "w") as f:
            yaml.dump(prometheus_config, f)

        # Alerts configuration
        alerts = []
        if alerting:
            alerts_config = {
                "groups": [
                    {
                        "name": "fraudlens",
                        "rules": [
                            {
                                "alert": "HighErrorRate",
                                "expr": "rate(fraudlens_errors_total[5m]) > 0.05",
                                "for": "5m",
                                "labels": {"severity": "critical"},
                                "annotations": {"summary": "High error rate detected"},
                            }
                        ],
                    }
                ]
            }

            alerts_path = self.deployment_dir / "alerts.yml"
            with open(alerts_path, "w") as f:
                yaml.dump(alerts_config, f)

            alerts = alerts_config["groups"][0]["rules"]

        return MonitoringDashboard(
            provider="prometheus",
            url=f"http://localhost:{metrics_port}",
            dashboards=["Targets", "Rules", "Alerts"],
            alerts=alerts,
            metrics_endpoint=f"http://localhost:{metrics_port}/metrics",
        )

    def generate_deployment_docs(self):
        """Generate comprehensive deployment documentation"""

        docs = """# FraudLens Deployment Guide

## Local Mac Deployment (M4 Optimized)

### Prerequisites
- macOS 13.0+
- Python 3.9+
- Xcode Command Line Tools
- Docker Desktop (optional)

### Quick Start
```bash
# Clone repository
git clone https://github.com/fraudlens/fraudlens.git
cd fraudlens

# Install dependencies
pip install -r requirements.txt

# Run optimization for M4
python -m fraudlens.optimization.optimizer --platform mac-m4

# Start server
python -m fraudlens.server
```

### Building Mac App
```bash
# Package as Mac app
python -m fraudlens.deployment.manager package-mac \\
    --sign-identity "Developer ID Application: Your Name"

# Install app
cp -r deployment/FraudLens.app /Applications/
```

## Docker Deployment

### Build Image
```bash
# Build multi-arch image
docker buildx build --platform linux/amd64,linux/arm64 \\
    -t fraudlens:latest .

# Run container
docker run -p 8000:8000 -p 7860:7860 fraudlens:latest
```

### Docker Compose
```bash
docker-compose up -d
```

## Kubernetes Deployment

### Prerequisites
- kubectl configured
- Helm 3+ (optional)

### Deploy with kubectl
```bash
# Apply manifests
kubectl apply -f deployment/k8s/

# Check status
kubectl get pods -n fraudlens
```

### Deploy with Helm
```bash
# Add repo
helm repo add fraudlens https://charts.fraudlens.io

# Install
helm install fraudlens fraudlens/fraudlens \\
    --namespace fraudlens --create-namespace
```

## Cloud Deployment

### AWS
```bash
# Deploy with CloudFormation
aws cloudformation create-stack \\
    --stack-name fraudlens \\
    --template-body file://deployment/aws/template.yaml

# Deploy with ECS
ecs-cli compose up --cluster fraudlens
```

### Google Cloud Platform
```bash
# Deploy to Cloud Run
gcloud run deploy fraudlens \\
    --image gcr.io/project/fraudlens \\
    --platform managed \\
    --allow-unauthenticated

# Deploy to GKE
gcloud container clusters create fraudlens \\
    --machine-type n2-standard-4 \\
    --num-nodes 3
kubectl apply -f deployment/k8s/
```

### Azure
```bash
# Deploy to Container Instances
az container create \\
    --resource-group fraudlens \\
    --name fraudlens \\
    --image fraudlens:latest \\
    --ports 8000 7860

# Deploy to AKS
az aks create \\
    --resource-group fraudlens \\
    --name fraudlens-cluster
kubectl apply -f deployment/k8s/
```

## Production Configuration

### Environment Variables
```bash
export FRAUDLENS_ENV=production
export FRAUDLENS_LOG_LEVEL=INFO
export FRAUDLENS_MAX_WORKERS=4
export FRAUDLENS_CACHE_SIZE=1000
export FRAUDLENS_MODEL_PATH=/models
```

### SSL/TLS Setup
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 \\
    -newkey rsa:2048 \\
    -keyout fraudlens.key \\
    -out fraudlens.crt

# Configure NGINX
cp deployment/nginx-ssl.conf /etc/nginx/sites-available/fraudlens
ln -s /etc/nginx/sites-available/fraudlens /etc/nginx/sites-enabled/
```

### Database Setup
```sql
CREATE DATABASE fraudlens;
CREATE USER fraudlens_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE fraudlens TO fraudlens_user;
```

## Monitoring

### Prometheus + Grafana
```bash
# Start Prometheus
prometheus --config.file=deployment/prometheus.yml

# Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/import \\
    -H "Content-Type: application/json" \\
    -d @deployment/grafana-dashboard.json
```

### Health Checks
- API Health: `GET /health`
- Metrics: `GET /metrics`
- Ready: `GET /ready`

## Scaling

### Horizontal Scaling
```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraudlens-hpa
spec:
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Vertical Scaling
```bash
# Increase resources
kubectl set resources deployment fraudlens \\
    --limits=cpu=4,memory=8Gi \\
    --requests=cpu=2,memory=4Gi
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable model quantization
   - Increase memory limits

2. **High Latency**
   - Enable caching
   - Use hardware acceleration
   - Optimize batch processing

3. **Model Loading Errors**
   - Check model path
   - Verify model format
   - Ensure sufficient permissions

### Debug Mode
```bash
export FRAUDLENS_DEBUG=true
export FRAUDLENS_LOG_LEVEL=DEBUG
python -m fraudlens.server --debug
```

## Support

- Documentation: https://docs.fraudlens.io
- Issues: https://github.com/fraudlens/fraudlens/issues
- Discord: https://discord.gg/fraudlens
"""

        docs_path = self.deployment_dir / "DEPLOYMENT.md"
        with open(docs_path, "w") as f:
            f.write(docs)

        print(f"✅ Deployment documentation generated: {docs_path}")
