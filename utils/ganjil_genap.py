def cek_ganjil_genap(plate_text):
    digits = ''.join(filter(str.isdigit, plate_text))
    if digits:
        last_digit = int(digits[-1])
        return 'Ganjil' if last_digit % 2 == 1 else 'Genap'
    return 'Tidak diketahui'
