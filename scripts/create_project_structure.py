import os

def create_directory_structure():
    # Define the project structure
    structure = {
        "data": {
            "raw": {},
            "processed": {}
        },
        "src": {
            "__init__.py": "",
            "data_collection": {
                "__init__.py": "",
                "scraper.py": "",
                "kafka_producer.py": "",
                "kinesis_producer.py": ""
            },
            "model": {
                "__init__.py": "",
                "price_model.py": "",
                "train_model.py": ""
            },
            "api": {
                "__init__.py": "",
                "app.py": "",
                "pricing_service.py": "",
                "redis_cache.py": ""
            },
            "dashboard": {
                "__init__.py": "",
                "dashboard.py": "",
                "api_integration.py": ""
            }
        },
        "config": {
            "kafka_config.py": "",
            "kinesis_config.py": "",
            "api_config.py": "",
            "model_config.py": ""
        },
        "scripts": {
            "data_cleaning.py": "",
            "start_kafka.py": "",
            "start_kinesis.py": ""
        },
        "requirements.txt": "",
        "Dockerfile": "",
        "README.md": "",
        ".gitignore": ""
    }

    def create_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, content)
            else:
                # Create empty file
                with open(path, 'w') as f:
                    pass

    # Create the structure
    create_structure(".", structure)

if __name__ == "__main__":
    create_directory_structure()
    print("Project structure created successfully!") 