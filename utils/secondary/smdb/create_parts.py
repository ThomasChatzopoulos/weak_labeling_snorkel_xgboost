import time

start_time = time.time()
path = "/corrections"
part = 1
lines = 0
batch = 0
with open(f'{path}/corr_{part}.sql', 'a', encoding="utf-8") as writer:
    writer.write("USE semmeddb43;LOCK TABLES `ENTITY` WRITE;")
with open(f"{path}/corrections.sql", encoding="utf-8") as file:
    for row in file:
        lines += 1
        # if lines >= 51:
        with open(f'{path}/corr_{part}.sql', 'a', encoding="utf-8") as writer:
            writer.write(row)
        batch += 1
        if batch == 20:
            part += 1
            batch = 0
            print(f"Part: {part} \t lines: {lines}")
            with open(f'{path}/corr_{part}.sql', 'a', encoding="utf-8") as writer:
                writer.write("USE semmeddb43;LOCK TABLES `ENTITY` WRITE;")

print("Total time:", time.time() - start_time)
