from sqlalchemy import Table, create_engine, MetaData, Column, Text, Integer, String, ForeignKeyConstraint, insert, select

class NimtsDocsItem:
    def __init__(self, filename:str, text:str, chunk_id:int):
        self.filename=filename
        self.text=text
        self.chunk_id = chunk_id

class DB_Helper():
    def __init__(self, db_name='nimts_rag'):
        self.meta = MetaData()
        self.nimts_docs = Table(
            'nimts_docs',
            self.meta,
            Column('filename', String, primary_key=True),
            Column('chunk_id', Integer, primary_key=True),
            Column('text', Text)
        )
        self.llm_summary = Table(
            'llm_summ',
            self.meta,
            Column('filename', String),
            Column('chunk_id', Integer),
            Column('summary', Text),
            ForeignKeyConstraint(
                ['filename', 'chunk_id'],
                ['nimts_docs.filename', 'nimts_docs.chunk_id']
            )
        )
        self.llm_total_summary = Table(
            'llm_total_summ',
            self.meta,
            Column('filename', String, primary_key=True),
            Column('total_summary', Text)
        )
        self.engine = create_engine(f'sqlite:///{db_name}.db')
        self.meta.create_all(self.engine)
        with self.engine.connect() as _:
            print('Successfully connected to SQLite.')
        
    def insert_nimts_docs(self, item: NimtsDocsItem):
        with self.engine.connect() as connection:
            prompt = insert(self.nimts_docs).values(filename=item.filename, text=item.text, chunk_id=item.chunk_id)
            connection.execute(prompt)
            connection.commit()
    
    def insert_llm_summ(self, item: NimtsDocsItem):
        with self.engine.connect() as connection:
            prompt = insert(self.llm_summary).values(filename=item.filename, summary=item.text, chunk_id=item.chunk_id)
            connection.execute(prompt)
            connection.commit()
    
    def select_nimts_docs(self, filename: str, chunk_id: int):
        with self.engine.connect() as connection:
            prompt = select(self.nimts_docs.c.text).where(
                self.nimts_docs.c.filename == filename,
                self.nimts_docs.c.chunk_id == chunk_id
            )
            result = connection.execute(prompt).fetchone()
            return result[0] if result else None    

    def select_all_nimts_chunkids_docs(self, filename: str):
        with self.engine.connect() as connection:
            prompt = select(self.nimts_docs.c.chunk_id).where(
                self.nimts_docs.c.filename == filename
            )
            results = connection.execute(prompt).fetchall()
            if not results:
                return set()
            return set(r[0] for r in results)

    def select_all_nimts_text_docs(self, filename: str):
        with self.engine.connect() as connection:
            prompt = select(self.nimts_docs.c.text).where(
                self.nimts_docs.c.filename == filename
            )
            results = connection.execute(prompt).fetchall()
            if not results:
                return None
            return "\n".join([row[0] for row in results])

    def select_all_nimts_filenames_docs(self):
        with self.engine.connect() as connection:
            prompt = select(self.nimts_docs.c.filename)
            results = connection.execute(prompt).fetchall()
            if not results:
                return set()
            return set(r[0] for r in results)


    def select_llm_summaries(self, filename: str, display_chunkid=True):
        with self.engine.connect() as connection:
            prompt = select(self.llm_summary.c.summary).where(
                self.llm_summary.c.filename == filename
            )
            results = connection.execute(prompt).fetchall()
            if not results:
                return None
            if not display_chunkid:
                return "\n".join([row[0]] for row in results)
            else:
                answer = ""
                for chunk, row in enumerate(results):
                    answer += f"\nCHUNK {chunk} ================== \n {row[0]}"
                return answer

    def select_all_llm_chunkids_summaries(self, filename: str):
        with self.engine.connect() as connection:
            prompt = select(self.llm_summary.c.chunk_id).where(
                self.llm_summary.c.filename == filename
            )
            results = connection.execute(prompt).fetchall()
            if not results:
                return set()
            return set(r[0] for r in results)

    def insert_total_llm_summary(self, filename: str, total_summary: str):
        with self.engine.connect() as connection:
            prompt = insert(self.llm_total_summary).values(filename=filename, total_summary=total_summary,)
            connection.execute(prompt)
            connection.commit()
    
    def select_total_llm_summary(self, filename: str):
        with self.engine.connect() as connection:
            prompt = select(self.llm_total_summary.c.total_summary).where(
                self.llm_total_summary.c.filename == filename
            )
            result = connection.execute(prompt).fetchone()
            return result[0] if result else None
    
    def select_total_summary_files(self):
        with self.engine.connect() as connection:
            prompt = select(self.llm_total_summary.c.filename)
            results = connection.execute(prompt).fetchall()
            if not results:
                return set()
            return set(r[0] for r in results)

    def select_total_llm_allfiles(self) -> dict:
        with self.engine.connect() as connection:
            prompt = select(self.llm_total_summary.c.filename, self.llm_total_summary.c.total_summary)
            results = connection.execute(prompt).fetchall()
            
            if not results:
                return None
            answer = {}
            for row in results:
                answer[row[0]] = row[1]
            return answer
