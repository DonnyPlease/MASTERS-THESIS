def log(log, message):
    with open(log, 'a') as f:
        f.write(message + '\n')
        