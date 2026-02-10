import sqlite3


class Database:
    def __init__(self):
        self.conn = sqlite3.connect("boulderform-scanner.db")
        self.cursor = self.conn.cursor()

        # Create competitions table if it does not exist yet
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS competitions (
            id INTEGER PRIMARY KEY,
            competition_name varchar,
            location varchar,
            competition_date DATE,
            category varchar,
            round varchar,
            retries INTEGER,
            amount_of_boulders INTEGER
        )""")
        # Create the participants table if it does not exist yet
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY,
            full_name varchar,
            first_name varchar,
            last_name varchar,
            lastname_prefix varchar,
            lastname_suffix varchar,
            display_last_name varchar,
            display_lastname_upcase varchar,
            gender char(1),
            birthdate DATE,
            city varchar,
            country char(60),
            team varchar,
            category varchar ,
            starting_number INTEGER,
            player_number INTEGER,
            exaequo INTEGER,
            countback INTEGER,
            dispensated INTEGER,
            outside_competition INTEGER,
            competition_id INTEGER,
            FOREIGN KEY(competition_id) REFERENCES competitions(id)
        )""")

        # Create a ResultsTable linking the participant and competition combo to the eventual results of the participants
        self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY,
                    serialized_results BLOB,
                    zone_total INTEGER,
                    tops_total INTEGER,
                    zone_tries INTEGER,
                    tops_tries INTEGER,
                    participant_id INTEGER,
                    competition_id INTEGER,
                    FOREIGN KEY(participant_id) REFERENCES participants(id),
                    FOREIGN KEY(competition_id) REFERENCES competitions(id)
                )""")

        self.conn.commit()


if __name__ == '__main__':
    db = Database()
