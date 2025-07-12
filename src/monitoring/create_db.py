from src.config.db_config import DATABASE_URI
from sqlalchemy import create_engine
from src.monitoring.utils.models import Base

if __name__ == "__main__":
    engine = create_engine(DATABASE_URI)
    Base.metadata.create_all(engine)
    print(f"Sucessfully created: {DATABASE_URI}")