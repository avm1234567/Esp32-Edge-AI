# Block 1: give each distinct character a numeric ID.
text = input("Enter some text: ")
vocabulary = sorted(set(text))

char_to_id = {}
for number, character in enumerate(vocabulary):
    char_to_id[character] = number

print("Text:", text)
print("Vocabulary:", vocabulary)
print("Character IDs:", char_to_id)
