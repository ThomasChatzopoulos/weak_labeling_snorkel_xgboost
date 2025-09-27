def seperate_inserts():
    path = "C:/temp/parts"
    with open(f"{path}/part_6336.sql") as file:
        with open(f"{path}/6336_seperated.sql", 'w') as writer:
            writer.write("USE semmeddb43;")
            for line in file:
                sp = line.split("),(")
                if sp[0] == 'USE semmeddb43;LOCK TABLES `ENTITY` WRITE;INSERT INTO `ENTITY` VALUES (122953395,6065300,\'3213163\',\'C0543467\',\'Operative Surgical Procedures\',\'topp\',\'\',\'\',\'surgery\',1000,241,248':
                    tp = sp[0].split("(")
                    sp[0] = tp[1]
                for row in sp:
                    line = f"INSERT INTO `ENTITY` VALUES({row});\n"
                    writer.write(line)


def rewriteErrorsCorrectEnconding():
    path = "C:/temp"
    smdb_path = "F:/PC_files/smdb/parts"

    with open(f"{path}/errors.txt", encoding="utf-8") as file:
        with open(f'{path}/corrections.sql', 'w', encoding="utf-8") as writer:
            writer.write("USE semmeddb43;LOCK TABLES `ENTITY` WRITE;\n")
            for line in file:
                print(line)
                if 'Incorrect string value:' in line:
                    spl_row = line.split('~')
                    lr = int(spl_row[0].split(':')[0].split("line")[1].replace(' ', ''))
                    with open(f"{smdb_path}/part_{spl_row[1].strip()}.sql") as smdb_file:
                        r = 0
                        for smdb_line in smdb_file:
                            r += 1
                            if lr == r:
                                print(f"in {r}")
                                writer.write(smdb_line)


seperate_inserts()
