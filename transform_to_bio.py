import re


def tag_entity(entity_type, entity_str):
    tagged = []
    entity_str = entity_str.replace(' ', '')
    for i, char in enumerate(entity_str):
        if i == 0:
            tagged.append(' '.join((char, 'B-' + entity_type)))
        else:
            tagged.append(' '.join((char, 'I-' + entity_type)))

    return '\n'.join(tagged)


def handle_one_line(line):
    new_line = []
    spans = re.finditer('{{(.*?)}}', line)
    next_span = next(spans, None)
    if next_span:
        last_position = 0
        while next_span:
            start, end = next_span.span()
            # handle previous Os
            if start > last_position:
                new_part = []
                for i in range(last_position, start):
                    new_part.append(' '.join((line[i], 'O')))
                new_line.append('\n'.join(new_part))
            # handle entities
            entity_content = next_span.group().strip('{').strip(
                '}').split(':')
            entity_type = entity_content[0]
            entity_str = ''.join(entity_content[1:])

            new_line.append(tag_entity(entity_type.strip(), entity_str.strip()))
            last_position = end
            next_span = next(spans, None)

        # handle last Os
        if last_position < len(line) - 1:
            new_part = []
            for i in range(last_position, len(line)):
                new_part.append(' '.join((line[i], 'O')))
            new_line.append('\n'.join(new_part))

    else:
        new_line = [' '.join((char, 'O')) for char in line]

    new_line = '\n'.join(new_line)
    return new_line


if __name__ == '__main__':
    with open('data/NER_corpus_chinese-master/Boson_NER_6C/origindata.txt') as f:
        new_lines = []
        for line in f.readlines():
            line = line.strip()
            line = line.replace(r'\n\n','')
            line = re.sub(r'\s+', ' ', line)
            new_line = handle_one_line(line)
            new_line = new_line.replace('。 O', '。 O\n')
            new_line = new_line.replace('！ O', '！ O\n')
            new_line = new_line.replace('？ O', '？ O\n')
            new_line = new_line.replace('～ O', '～ O\n')
            new_line = new_line.replace('~ O', '~ O\n')
            new_line = new_line.replace('； O', '； O\n')
            new_line = new_line.replace('\n\n\n', '\n\n')
            new_lines.append(new_line)

        new_lines = '\n\n'.join(new_lines)
        new_lines = new_lines.replace('\n\n\n', '\n\n')

    with open('data/Boson.txt','w') as f:
        f.write(new_lines)
