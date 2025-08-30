#!/bin/bash

# FraudLens Deployment Script
# Automates deployment to Docker, Kubernetes, or Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fraudlens"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"docker.io"}
DOCKER_IMAGE=${DOCKER_IMAGE:-"fraudlens"}
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed"
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl is not installed"
    fi
    
    print_info "Prerequisites check completed"
}

# Function to load environment variables
load_env() {
    if [ -f ".env.${ENVIRONMENT}" ]; then
        print_info "Loading environment: ${ENVIRONMENT}"
        export $(cat .env.${ENVIRONMENT} | grep -v '^#' | xargs)
    else
        print_warning "Environment file .env.${ENVIRONMENT} not found"
    fi
}

# Function to build Docker image
build_docker() {
    print_info "Building Docker image..."
    
    BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" \
        -f Dockerfile.production \
        -t ${DOCKER_IMAGE}:${VERSION} \
        -t ${DOCKER_IMAGE}:latest \
        .
    
    print_info "Docker image built successfully"
}

# Function to push Docker image
push_docker() {
    print_info "Pushing Docker image to registry..."
    
    docker tag ${DOCKER_IMAGE}:${VERSION} ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${VERSION}
    docker tag ${DOCKER_IMAGE}:latest ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest
    
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${VERSION}
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest
    
    print_info "Docker image pushed successfully"
}

# Function to deploy with Docker Compose
deploy_compose() {
    print_info "Deploying with Docker Compose..."
    
    COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"
    if [ ! -f "${COMPOSE_FILE}" ]; then
        COMPOSE_FILE="docker-compose.production.yml"
    fi
    
    docker-compose -f ${COMPOSE_FILE} up -d
    
    print_info "Docker Compose deployment completed"
    print_info "Services:"
    docker-compose -f ${COMPOSE_FILE} ps
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    print_info "Deploying to Kubernetes..."
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        print_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    # Apply base configuration
    kubectl apply -k k8s/base/
    
    # Apply environment-specific overlay if exists
    if [ -d "k8s/overlays/${ENVIRONMENT}" ]; then
        print_info "Applying ${ENVIRONMENT} overlay..."
        kubectl apply -k k8s/overlays/${ENVIRONMENT}/
    fi
    
    # Wait for deployment
    print_info "Waiting for deployment to be ready..."
    kubectl -n fraudlens wait --for=condition=available --timeout=300s deployment/fraudlens-api
    
    print_info "Kubernetes deployment completed"
    kubectl -n fraudlens get pods
}

# Function to run database migrations
run_migrations() {
    print_info "Running database migrations..."
    
    docker run --rm \
        --env-file .env.${ENVIRONMENT} \
        ${DOCKER_IMAGE}:${VERSION} \
        python -m fraudlens.db.migrate
    
    print_info "Migrations completed"
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    
    docker run --rm \
        --env-file .env.${ENVIRONMENT} \
        ${DOCKER_IMAGE}:${VERSION} \
        python -m pytest tests/
    
    print_info "Tests completed"
}

# Function to health check
health_check() {
    print_info "Performing health check..."
    
    # Try to get the service URL
    if [ "$DEPLOYMENT_TYPE" == "compose" ]; then
        URL="http://localhost:8000/health"
    elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        URL=$(kubectl -n fraudlens get ingress fraudlens-ingress -o jsonpath='{.spec.rules[0].host}')
        URL="https://${URL}/health"
    else
        URL="http://localhost:8000/health"
    fi
    
    # Wait for service to be ready
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f ${URL} &> /dev/null; then
            print_info "Health check passed"
            break
        else
            print_warning "Service not ready, retrying... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep 10
            RETRY_COUNT=$((RETRY_COUNT + 1))
        fi
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Health check failed after $MAX_RETRIES attempts"
        exit 1
    fi
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Set environment (development|staging|production)"
    echo "  -v, --version VERSION    Set version tag"
    echo "  -t, --type TYPE         Deployment type (compose|kubernetes|docker)"
    echo "  -b, --build             Build Docker image"
    echo "  -p, --push              Push to registry"
    echo "  -m, --migrate           Run database migrations"
    echo "  -T, --test              Run tests"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e production -t compose -b     # Build and deploy with Docker Compose"
    echo "  $0 -e production -t kubernetes -bp  # Build, push, and deploy to Kubernetes"
}

# Parse command line arguments
DEPLOYMENT_TYPE=""
BUILD_IMAGE=false
PUSH_IMAGE=false
RUN_MIGRATIONS=false
RUN_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_IMAGE=true
            shift
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -m|--migrate)
            RUN_MIGRATIONS=true
            shift
            ;;
        -T|--test)
            RUN_TESTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main deployment flow
print_info "Starting FraudLens deployment"
print_info "Environment: ${ENVIRONMENT}"
print_info "Version: ${VERSION}"

# Check prerequisites
check_prerequisites

# Load environment
load_env

# Build if requested
if [ "$BUILD_IMAGE" = true ]; then
    build_docker
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    run_tests
fi

# Push if requested
if [ "$PUSH_IMAGE" = true ]; then
    push_docker
fi

# Run migrations if requested
if [ "$RUN_MIGRATIONS" = true ]; then
    run_migrations
fi

# Deploy based on type
case $DEPLOYMENT_TYPE in
    compose)
        deploy_compose
        ;;
    kubernetes|k8s)
        deploy_kubernetes
        ;;
    docker)
        print_info "Starting Docker container..."
        docker run -d \
            --name ${PROJECT_NAME} \
            --env-file .env.${ENVIRONMENT} \
            -p 8000:8000 \
            ${DOCKER_IMAGE}:${VERSION}
        ;;
    *)
        print_warning "No deployment type specified, skipping deployment"
        ;;
esac

# Health check
if [ -n "$DEPLOYMENT_TYPE" ]; then
    health_check
fi

print_info "Deployment completed successfully!"