from configparser import ConfigParser
from ast import literal_eval as safe_eval


class SafeConfigParser(ConfigParser):
    """
    This class was create to safely parse and evaluate config files.
    Using functions, which end in _evaluated will return an evaluated python literal
    or a safe string.
    """

    # suppress all system calls
    harmful_calls = ['os.',
                     'sys.',
                     ]

    def __init__(self):
        super(SafeConfigParser, self).__init__()

    def items_evaluated(self, section, *args, **kwargs):
        section_items = dict(self.items(section, *args, *kwargs))
        evaluated_items = dict()
        for k, v in section_items.items():
            try:
                evaluated_items[k] = safe_eval(v)
            except ValueError:
                # keep as str
                evaluated_items[k] = self.safety_check(literal=v)
            except SyntaxError:
                # path variables do not comply with python syntax
                evaluated_items[k] = self.safety_check(literal=v)
        return evaluated_items

    def get_evaluated(self, section, option, *args, **kwargs):
        element = self.get(section, option, *args, **kwargs)
        try:
            return safe_eval(element)
        except ValueError:
            # keep as str
            return self.safety_check(literal=element)
        except SyntaxError:
            # path variables do not comply with python syntax
            return self.safety_check(literal=element)

    def safety_check(self, literal):
        if any([(c in literal) for c in self.harmful_calls]):
            raise ValueError("Harmful literal found in config!")
        return literal





