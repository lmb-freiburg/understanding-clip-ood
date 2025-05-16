import json
import os
from typing import Callable, Sequence

import numpy as np
import torch
import tqdm
from PIL import Image
from textacy import preprocessing
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder

openai_imagenet_classes = [
    'tench',
    'goldfish',
    'great white shark',
    'tiger shark',
    'hammerhead shark',
    'electric ray',
    'stingray',
    'rooster',
    'hen',
    'ostrich',
    'brambling',
    'goldfinch',
    'house finch',
    'junco',
    'indigo bunting',
    'American robin',
    'bulbul',
    'jay',
    'magpie',
    'chickadee',
    'American dipper',
    'kite (bird of prey)',
    'bald eagle',
    'vulture',
    'great grey owl',
    'fire salamander',
    'smooth newt',
    'eft',
    'spotted salamander',
    'axolotl',
    'American bullfrog',
    'tree frog',
    'tailed frog',
    'loggerhead sea turtle',
    'leatherback sea turtle',
    'mud turtle',
    'terrapin',
    'box turtle',
    'banded gecko',
    'green iguana',
    'Carolina anole',
    'desert grassland whiptail lizard',
    'agama',
    'frilled-necked lizard',
    'alligator lizard',
    'Gila monster',
    'European green lizard',
    'chameleon',
    'Komodo dragon',
    'Nile crocodile',
    'American alligator',
    'triceratops',
    'worm snake',
    'ring-necked snake',
    'eastern hog-nosed snake',
    'smooth green snake',
    'kingsnake',
    'garter snake',
    'water snake',
    'vine snake',
    'night snake',
    'boa constrictor',
    'African rock python',
    'Indian cobra',
    'green mamba',
    'sea snake',
    'Saharan horned viper',
    'eastern diamondback rattlesnake',
    'sidewinder rattlesnake',
    'trilobite',
    'harvestman',
    'scorpion',
    'yellow garden spider',
    'barn spider',
    'European garden spider',
    'southern black widow',
    'tarantula',
    'wolf spider',
    'tick',
    'centipede',
    'black grouse',
    'ptarmigan',
    'ruffed grouse',
    'prairie grouse',
    'peafowl',
    'quail',
    'partridge',
    'african grey parrot',
    'macaw',
    'sulphur-crested cockatoo',
    'lorikeet',
    'coucal',
    'bee eater',
    'hornbill',
    'hummingbird',
    'jacamar',
    'toucan',
    'duck',
    'red-breasted merganser',
    'goose',
    'black swan',
    'tusker',
    'echidna',
    'platypus',
    'wallaby',
    'koala',
    'wombat',
    'jellyfish',
    'sea anemone',
    'brain coral',
    'flatworm',
    'nematode',
    'conch',
    'snail',
    'slug',
    'sea slug',
    'chiton',
    'chambered nautilus',
    'Dungeness crab',
    'rock crab',
    'fiddler crab',
    'red king crab',
    'American lobster',
    'spiny lobster',
    'crayfish',
    'hermit crab',
    'isopod',
    'white stork',
    'black stork',
    'spoonbill',
    'flamingo',
    'little blue heron',
    'great egret',
    'bittern bird',
    'crane bird',
    'limpkin',
    'common gallinule',
    'American coot',
    'bustard',
    'ruddy turnstone',
    'dunlin',
    'common redshank',
    'dowitcher',
    'oystercatcher',
    'pelican',
    'king penguin',
    'albatross',
    'grey whale',
    'killer whale',
    'dugong',
    'sea lion',
    'Chihuahua',
    'Japanese Chin',
    'Maltese',
    'Pekingese',
    'Shih Tzu',
    'King Charles Spaniel',
    'Papillon',
    'toy terrier',
    'Rhodesian Ridgeback',
    'Afghan Hound',
    'Basset Hound',
    'Beagle',
    'Bloodhound',
    'Bluetick Coonhound',
    'Black and Tan Coonhound',
    'Treeing Walker Coonhound',
    'English foxhound',
    'Redbone Coonhound',
    'borzoi',
    'Irish Wolfhound',
    'Italian Greyhound',
    'Whippet',
    'Ibizan Hound',
    'Norwegian Elkhound',
    'Otterhound',
    'Saluki',
    'Scottish Deerhound',
    'Weimaraner',
    'Staffordshire Bull Terrier',
    'American Staffordshire Terrier',
    'Bedlington Terrier',
    'Border Terrier',
    'Kerry Blue Terrier',
    'Irish Terrier',
    'Norfolk Terrier',
    'Norwich Terrier',
    'Yorkshire Terrier',
    'Wire Fox Terrier',
    'Lakeland Terrier',
    'Sealyham Terrier',
    'Airedale Terrier',
    'Cairn Terrier',
    'Australian Terrier',
    'Dandie Dinmont Terrier',
    'Boston Terrier',
    'Miniature Schnauzer',
    'Giant Schnauzer',
    'Standard Schnauzer',
    'Scottish Terrier',
    'Tibetan Terrier',
    'Australian Silky Terrier',
    'Soft-coated Wheaten Terrier',
    'West Highland White Terrier',
    'Lhasa Apso',
    'Flat-Coated Retriever',
    'Curly-coated Retriever',
    'Golden Retriever',
    'Labrador Retriever',
    'Chesapeake Bay Retriever',
    'German Shorthaired Pointer',
    'Vizsla',
    'English Setter',
    'Irish Setter',
    'Gordon Setter',
    'Brittany dog',
    'Clumber Spaniel',
    'English Springer Spaniel',
    'Welsh Springer Spaniel',
    'Cocker Spaniel',
    'Sussex Spaniel',
    'Irish Water Spaniel',
    'Kuvasz',
    'Schipperke',
    'Groenendael dog',
    'Malinois',
    'Briard',
    'Australian Kelpie',
    'Komondor',
    'Old English Sheepdog',
    'Shetland Sheepdog',
    'collie',
    'Border Collie',
    'Bouvier des Flandres dog',
    'Rottweiler',
    'German Shepherd Dog',
    'Dobermann',
    'Miniature Pinscher',
    'Greater Swiss Mountain Dog',
    'Bernese Mountain Dog',
    'Appenzeller Sennenhund',
    'Entlebucher Sennenhund',
    'Boxer',
    'Bullmastiff',
    'Tibetan Mastiff',
    'French Bulldog',
    'Great Dane',
    'St. Bernard',
    'husky',
    'Alaskan Malamute',
    'Siberian Husky',
    'Dalmatian',
    'Affenpinscher',
    'Basenji',
    'pug',
    'Leonberger',
    'Newfoundland dog',
    'Great Pyrenees dog',
    'Samoyed',
    'Pomeranian',
    'Chow Chow',
    'Keeshond',
    'brussels griffon',
    'Pembroke Welsh Corgi',
    'Cardigan Welsh Corgi',
    'Toy Poodle',
    'Miniature Poodle',
    'Standard Poodle',
    'Mexican hairless dog (xoloitzcuintli)',
    'grey wolf',
    'Alaskan tundra wolf',
    'red wolf or maned wolf',
    'coyote',
    'dingo',
    'dhole',
    'African wild dog',
    'hyena',
    'red fox',
    'kit fox',
    'Arctic fox',
    'grey fox',
    'tabby cat',
    'tiger cat',
    'Persian cat',
    'Siamese cat',
    'Egyptian Mau',
    'cougar',
    'lynx',
    'leopard',
    'snow leopard',
    'jaguar',
    'lion',
    'tiger',
    'cheetah',
    'brown bear',
    'American black bear',
    'polar bear',
    'sloth bear',
    'mongoose',
    'meerkat',
    'tiger beetle',
    'ladybug',
    'ground beetle',
    'longhorn beetle',
    'leaf beetle',
    'dung beetle',
    'rhinoceros beetle',
    'weevil',
    'fly',
    'bee',
    'ant',
    'grasshopper',
    'cricket insect',
    'stick insect',
    'cockroach',
    'praying mantis',
    'cicada',
    'leafhopper',
    'lacewing',
    'dragonfly',
    'damselfly',
    'red admiral butterfly',
    'ringlet butterfly',
    'monarch butterfly',
    'small white butterfly',
    'sulphur butterfly',
    'gossamer-winged butterfly',
    'starfish',
    'sea urchin',
    'sea cucumber',
    'cottontail rabbit',
    'hare',
    'Angora rabbit',
    'hamster',
    'porcupine',
    'fox squirrel',
    'marmot',
    'beaver',
    'guinea pig',
    'common sorrel horse',
    'zebra',
    'pig',
    'wild boar',
    'warthog',
    'hippopotamus',
    'ox',
    'water buffalo',
    'bison',
    'ram (adult male sheep)',
    'bighorn sheep',
    'Alpine ibex',
    'hartebeest',
    'impala (antelope)',
    'gazelle',
    'arabian camel',
    'llama',
    'weasel',
    'mink',
    'European polecat',
    'black-footed ferret',
    'otter',
    'skunk',
    'badger',
    'armadillo',
    'three-toed sloth',
    'orangutan',
    'gorilla',
    'chimpanzee',
    'gibbon',
    'siamang',
    'guenon',
    'patas monkey',
    'baboon',
    'macaque',
    'langur',
    'black-and-white colobus',
    'proboscis monkey',
    'marmoset',
    'white-headed capuchin',
    'howler monkey',
    'titi monkey',
    "Geoffroy's spider monkey",
    'common squirrel monkey',
    'ring-tailed lemur',
    'indri',
    'Asian elephant',
    'African bush elephant',
    'red panda',
    'giant panda',
    'snoek fish',
    'eel',
    'silver salmon',
    'rock beauty fish',
    'clownfish',
    'sturgeon',
    'gar fish',
    'lionfish',
    'pufferfish',
    'abacus',
    'abaya',
    'academic gown',
    'accordion',
    'acoustic guitar',
    'aircraft carrier',
    'airliner',
    'airship',
    'altar',
    'ambulance',
    'amphibious vehicle',
    'analog clock',
    'apiary',
    'apron',
    'trash can',
    'assault rifle',
    'backpack',
    'bakery',
    'balance beam',
    'balloon',
    'ballpoint pen',
    'Band-Aid',
    'banjo',
    'baluster / handrail',
    'barbell',
    'barber chair',
    'barbershop',
    'barn',
    'barometer',
    'barrel',
    'wheelbarrow',
    'baseball',
    'basketball',
    'bassinet',
    'bassoon',
    'swimming cap',
    'bath towel',
    'bathtub',
    'station wagon',
    'lighthouse',
    'beaker',
    'military hat (bearskin or shako)',
    'beer bottle',
    'beer glass',
    'bell tower',
    'baby bib',
    'tandem bicycle',
    'bikini',
    'ring binder',
    'binoculars',
    'birdhouse',
    'boathouse',
    'bobsleigh',
    'bolo tie',
    'poke bonnet',
    'bookcase',
    'bookstore',
    'bottle cap',
    'hunting bow',
    'bow tie',
    'brass memorial plaque',
    'bra',
    'breakwater',
    'breastplate',
    'broom',
    'bucket',
    'buckle',
    'bulletproof vest',
    'high-speed train',
    'butcher shop',
    'taxicab',
    'cauldron',
    'candle',
    'cannon',
    'canoe',
    'can opener',
    'cardigan',
    'car mirror',
    'carousel',
    'tool kit',
    'cardboard box / carton',
    'car wheel',
    'automated teller machine',
    'cassette',
    'cassette player',
    'castle',
    'catamaran',
    'CD player',
    'cello',
    'mobile phone',
    'chain',
    'chain-link fence',
    'chain mail',
    'chainsaw',
    'storage chest',
    'chiffonier',
    'bell or wind chime',
    'china cabinet',
    'Christmas stocking',
    'church',
    'movie theater',
    'cleaver',
    'cliff dwelling',
    'cloak',
    'clogs',
    'cocktail shaker',
    'coffee mug',
    'coffeemaker',
    'spiral or coil',
    'combination lock',
    'computer keyboard',
    'candy store',
    'container ship',
    'convertible',
    'corkscrew',
    'cornet',
    'cowboy boot',
    'cowboy hat',
    'cradle',
    'construction crane',
    'crash helmet',
    'crate',
    'infant bed',
    'Crock Pot',
    'croquet ball',
    'crutch',
    'cuirass',
    'dam',
    'desk',
    'desktop computer',
    'rotary dial telephone',
    'diaper',
    'digital clock',
    'digital watch',
    'dining table',
    'dishcloth',
    'dishwasher',
    'disc brake',
    'dock',
    'dog sled',
    'dome',
    'doormat',
    'drilling rig',
    'drum',
    'drumstick',
    'dumbbell',
    'Dutch oven',
    'electric fan',
    'electric guitar',
    'electric locomotive',
    'entertainment center',
    'envelope',
    'espresso machine',
    'face powder',
    'feather boa',
    'filing cabinet',
    'fireboat',
    'fire truck',
    'fire screen',
    'flagpole',
    'flute',
    'folding chair',
    'football helmet',
    'forklift',
    'fountain',
    'fountain pen',
    'four-poster bed',
    'freight car',
    'French horn',
    'frying pan',
    'fur coat',
    'garbage truck',
    'gas mask or respirator',
    'gas pump',
    'goblet',
    'go-kart',
    'golf ball',
    'golf cart',
    'gondola',
    'gong',
    'gown',
    'grand piano',
    'greenhouse',
    'radiator grille',
    'grocery store',
    'guillotine',
    'hair clip',
    'hair spray',
    'half-track',
    'hammer',
    'hamper',
    'hair dryer',
    'hand-held computer',
    'handkerchief',
    'hard disk drive',
    'harmonica',
    'harp',
    'combine harvester',
    'hatchet',
    'holster',
    'home theater',
    'honeycomb',
    'hook',
    'hoop skirt',
    'gymnastic horizontal bar',
    'horse-drawn vehicle',
    'hourglass',
    'iPod',
    'clothes iron',
    'carved pumpkin',
    'jeans',
    'jeep',
    'T-shirt',
    'jigsaw puzzle',
    'rickshaw',
    'joystick',
    'kimono',
    'knee pad',
    'knot',
    'lab coat',
    'ladle',
    'lampshade',
    'laptop computer',
    'lawn mower',
    'lens cap',
    'letter opener',
    'library',
    'lifeboat',
    'lighter',
    'limousine',
    'ocean liner',
    'lipstick',
    'slip-on shoe',
    'lotion',
    'music speaker',
    'loupe magnifying glass',
    'sawmill',
    'magnetic compass',
    'messenger bag',
    'mailbox',
    'maillot',
    'one-piece bathing suit',
    'manhole cover',
    'maraca',
    'marimba',
    'mask',
    'matchstick',
    'maypole',
    'maze',
    'measuring cup',
    'medicine cabinet',
    'megalith',
    'microphone',
    'microwave oven',
    'military uniform',
    'milk can',
    'minibus',
    'miniskirt',
    'minivan',
    'missile',
    'mitten',
    'mixing bowl',
    'mobile home',
    'ford model t',
    'modem',
    'monastery',
    'monitor',
    'moped',
    'mortar and pestle',
    'graduation cap',
    'mosque',
    'mosquito net',
    'vespa',
    'mountain bike',
    'tent',
    'computer mouse',
    'mousetrap',
    'moving van',
    'muzzle',
    'metal nail',
    'neck brace',
    'necklace',
    'baby pacifier',
    'notebook computer',
    'obelisk',
    'oboe',
    'ocarina',
    'odometer',
    'oil filter',
    'pipe organ',
    'oscilloscope',
    'overskirt',
    'bullock cart',
    'oxygen mask',
    'product packet / packaging',
    'paddle',
    'paddle wheel',
    'padlock',
    'paintbrush',
    'pajamas',
    'palace',
    'pan flute',
    'paper towel',
    'parachute',
    'parallel bars',
    'park bench',
    'parking meter',
    'railroad car',
    'patio',
    'payphone',
    'pedestal',
    'pencil case',
    'pencil sharpener',
    'perfume',
    'Petri dish',
    'photocopier',
    'plectrum',
    'Pickelhaube',
    'picket fence',
    'pickup truck',
    'pier',
    'piggy bank',
    'pill bottle',
    'pillow',
    'ping-pong ball',
    'pinwheel',
    'pirate ship',
    'drink pitcher',
    'block plane',
    'planetarium',
    'plastic bag',
    'plate rack',
    'farm plow',
    'plunger',
    'Polaroid camera',
    'pole',
    'police van',
    'poncho',
    'pool table',
    'soda bottle',
    'plant pot',
    "potter's wheel",
    'power drill',
    'prayer rug',
    'printer',
    'prison',
    'projectile',
    'projector',
    'hockey puck',
    'punching bag',
    'purse',
    'quill',
    'quilt',
    'race car',
    'racket',
    'radiator',
    'radio',
    'radio telescope',
    'rain barrel',
    'recreational vehicle',
    'fishing casting reel',
    'reflex camera',
    'refrigerator',
    'remote control',
    'restaurant',
    'revolver',
    'rifle',
    'rocking chair',
    'rotisserie',
    'eraser',
    'rugby ball',
    'ruler measuring stick',
    'sneaker',
    'safe',
    'safety pin',
    'salt shaker',
    'sandal',
    'sarong',
    'saxophone',
    'scabbard',
    'weighing scale',
    'school bus',
    'schooner',
    'scoreboard',
    'CRT monitor',
    'screw',
    'screwdriver',
    'seat belt',
    'sewing machine',
    'shield',
    'shoe store',
    'shoji screen / room divider',
    'shopping basket',
    'shopping cart',
    'shovel',
    'shower cap',
    'shower curtain',
    'ski',
    'balaclava ski mask',
    'sleeping bag',
    'slide rule',
    'sliding door',
    'slot machine',
    'snorkel',
    'snowmobile',
    'snowplow',
    'soap dispenser',
    'soccer ball',
    'sock',
    'solar thermal collector',
    'sombrero',
    'soup bowl',
    'keyboard space bar',
    'space heater',
    'space shuttle',
    'spatula',
    'motorboat',
    'spider web',
    'spindle',
    'sports car',
    'spotlight',
    'stage',
    'steam locomotive',
    'through arch bridge',
    'steel drum',
    'stethoscope',
    'scarf',
    'stone wall',
    'stopwatch',
    'stove',
    'strainer',
    'tram',
    'stretcher',
    'couch',
    'stupa',
    'submarine',
    'suit',
    'sundial',
    'sunglass',
    'sunglasses',
    'sunscreen',
    'suspension bridge',
    'mop',
    'sweatshirt',
    'swim trunks / shorts',
    'swing',
    'electrical switch',
    'syringe',
    'table lamp',
    'tank',
    'tape player',
    'teapot',
    'teddy bear',
    'television',
    'tennis ball',
    'thatched roof',
    'front curtain',
    'thimble',
    'threshing machine',
    'throne',
    'tile roof',
    'toaster',
    'tobacco shop',
    'toilet seat',
    'torch',
    'totem pole',
    'tow truck',
    'toy store',
    'tractor',
    'semi-trailer truck',
    'tray',
    'trench coat',
    'tricycle',
    'trimaran',
    'tripod',
    'triumphal arch',
    'trolleybus',
    'trombone',
    'hot tub',
    'turnstile',
    'typewriter keyboard',
    'umbrella',
    'unicycle',
    'upright piano',
    'vacuum cleaner',
    'vase',
    'vaulted or arched ceiling',
    'velvet fabric',
    'vending machine',
    'vestment',
    'viaduct',
    'violin',
    'volleyball',
    'waffle iron',
    'wall clock',
    'wallet',
    'wardrobe',
    'military aircraft',
    'sink',
    'washing machine',
    'water bottle',
    'water jug',
    'water tower',
    'whiskey jug',
    'whistle',
    'hair wig',
    'window screen',
    'window shade',
    'Windsor tie',
    'wine bottle',
    'airplane wing',
    'wok',
    'wooden spoon',
    'wool',
    'split-rail fence',
    'shipwreck',
    'sailboat',
    'yurt',
    'website',
    'comic book',
    'crossword',
    'traffic or street sign',
    'traffic light',
    'dust jacket',
    'menu',
    'plate',
    'guacamole',
    'consomme',
    'hot pot',
    'trifle',
    'ice cream',
    'popsicle',
    'baguette',
    'bagel',
    'pretzel',
    'cheeseburger',
    'hot dog',
    'mashed potatoes',
    'cabbage',
    'broccoli',
    'cauliflower',
    'zucchini',
    'spaghetti squash',
    'acorn squash',
    'butternut squash',
    'cucumber',
    'artichoke',
    'bell pepper',
    'cardoon',
    'mushroom',
    'Granny Smith apple',
    'strawberry',
    'orange',
    'lemon',
    'fig',
    'pineapple',
    'banana',
    'jackfruit',
    'cherimoya (custard apple)',
    'pomegranate',
    'hay',
    'carbonara',
    'chocolate syrup',
    'dough',
    'meatloaf',
    'pizza',
    'pot pie',
    'burrito',
    'red wine',
    'espresso',
    'tea cup',
    'eggnog',
    'mountain',
    'bubble',
    'cliff',
    'coral reef',
    'geyser',
    'lakeshore',
    'promontory',
    'sandbar',
    'beach',
    'valley',
    'volcano',
    'baseball player',
    'bridegroom',
    'scuba diver',
    'rapeseed',
    'daisy',
    "yellow lady's slipper",
    'corn',
    'acorn',
    'rose hip',
    'horse chestnut seed',
    'coral fungus',
    'agaric',
    'gyromitra',
    'stinkhorn mushroom',
    'earth star fungus',
    'hen of the woods mushroom',
    'bolete',
    'corn cob',
    'toilet paper',
]


class ImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        class_idcs: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        _ = kwargs  # Just for consistency with other datasets.
        assert split in ['train', 'val']
        path = os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)

        self.class_labels = {i: class_name for i, class_name in enumerate(openai_imagenet_classes)}
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}
            self.class_labels = {tgt_to_tgt_map[k]: v for k, v in self.class_labels.items() if k in tgt_to_tgt_map}

        self.targets = np.array(self.samples)[:, 1]


class CorruptedImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int = 3,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        class_idcs: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, corruption)
        assert os.path.isdir(path)
        path = os.path.join(path, str(severity))
        assert os.path.isdir(path)
        super().__init__(path, transform=transform, target_transform=target_transform)

        self.class_labels = {i: class_name for i, class_name in enumerate(openai_imagenet_classes)}
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}
            self.class_labels = {tgt_to_tgt_map[k]: v for k, v in self.class_labels.items() if k in tgt_to_tgt_map}

        self.targets = np.array(self.samples)[:, 1]


class ImageNetSketch(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str = 'sketch',
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        class_idcs: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        _ = kwargs  # Just for consistency with other datasets.
        assert split in ['sketch']
        path = os.path.join(root, split) if not root.endswith('sketch') else root
        super().__init__(path, transform=transform, target_transform=target_transform)

        self.class_labels = {i: class_name for i, class_name in enumerate(openai_imagenet_classes)}
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}
            self.class_labels = {tgt_to_tgt_map[k]: v for k, v in self.class_labels.items() if k in tgt_to_tgt_map}

        self.targets = np.array(self.samples)[:, 1]


class ImageNetCaptions(Dataset):
    def __init__(
        self,
        shard_path: str,
        imagenet_path: str,
        split: str,
        transform: Callable,
        target_transform: Callable | None = None,
        mode: str = 'label',
    ) -> None:
        shard_path = os.path.abspath(shard_path)
        imagenet_path = os.path.abspath(imagenet_path)
        assert all([os.path.isdir(os.path.join(imagenet_path, split)) for split in ['train', 'sketch', 'captions']])
        self.class_to_idx = ImageNet(imagenet_path, 'train').class_to_idx

        with open(shard_path) as f:
            json_data = json.load(f)
        if any(split in json_data for split in ['train', 'val']):
            assert split in ['train', 'val']
            shards = json_data[split]
            img_paths = [path for shard in shards for path in shard]
            img_labels = [self.class_to_idx[self._wnid_from_path(path)] for path in img_paths]
        else:
            img_paths = [os.path.join('captions', data['wnid'], data['filename']) for data in json_data]
            img_labels = [self.class_to_idx[data['wnid']] for data in json_data]
        self.samples = [(os.path.join(imagenet_path, path), label) for path, label in zip(img_paths, img_labels)]

        assert mode in ['label', 'caption', 'label+caption', 'path']
        self.return_label = 'label' in mode
        self.return_caption = 'caption' in mode
        self.return_path = 'path' in mode

        self.transform = transform
        self.target_transform = target_transform

    def _wnid_from_path(self, path: str) -> str:
        _, wnid, _ = path.split('/')
        assert len(wnid) == 9
        return wnid

    def _caption_from_path(self, path: str) -> str:
        with open(f'{os.path.splitext(path)[0]}.json') as f:
            caption = json.load(f)['caption']
        return caption

    def to_tsv(self, path: str, disable_pbar: bool = False, preprocess_text: bool = True) -> None:
        with open(path, 'w') as f:
            f.write('filepath\ttitle\n')

            for path, _ in tqdm.tqdm(self.samples, desc='Creating tsv', disable=disable_pbar):
                caption = self._caption_from_path(path)
                caption = caption.replace('\n', ' ')  # this is needed as in tsv one line equals one sample
                if preprocess_text:
                    caption = preprocessing.remove.html_tags(caption)
                f.write('\t'.join([path, caption]) + '\n')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        img = self.transform(Image.open(path).convert('RGB'))

        if self.target_transform:
            label = self.target_transform(label)

        if self.return_path:
            return img, path

        sample = (img, label) if self.return_label else (img,)
        if self.return_caption:
            sample += (self._caption_from_path(path),)

        return sample


class DomainNetCaptions(Dataset):
    def __init__(
        self,
        domainnet_path: str,
        split: str,
        transform: Callable,
        exclude_domains: list[str] = [],
        filter_classes: dict[str, set[int]] = {},
        mode: str = 'label',
    ) -> None:
        domainnet_path = os.path.abspath(domainnet_path)

        assert split in ['train', 'val']
        split = 'test' if split == 'val' else split
        assert mode in ['none', 'label', 'caption', 'label+caption']
        self.return_label = 'label' in mode
        self.return_caption = 'caption' in mode

        self.samples_per_domain = {'clipart': 0, 'infograph': 0, 'painting': 0, 'quickdraw': 0, 'real': 0, 'sketch': 0}
        self.samples = []
        for domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
            if domain in exclude_domains:
                continue

            with open(os.path.join(domainnet_path, f'{domain}_{split}.tsv')) as f:
                samples = f.readlines()

            samples = [sample.split('\t') for sample in samples]
            samples = [
                (os.path.join(domainnet_path, path), int(label), caption.strip()) for path, label, caption in samples
            ]

            # filter out certain classes for certain domains
            if domain in filter_classes:
                samples = [sample for sample in samples if sample[1] not in filter_classes[domain]]

            self.samples_per_domain[domain] = len(samples)
            self.samples.extend(samples)

        self.transform = transform

    def to_tsv(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write('filepath\ttitle\n')
            f.writelines(['\t'.join([path, caption]) + '\n' for path, _, caption in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple | str:
        path, label, caption = self.samples[index]
        img = self.transform(Image.open(path))

        # resolve return values
        sample = (img, label) if self.return_label else (img,)
        sample += (caption,) if self.return_caption else ()
        assert len(sample) > 0
        return sample if len(sample) > 1 else sample[0]


class TsvDataset(Dataset):
    def __init__(
        self, tsv_path: str, img_transform: Callable, txt_transform: Callable | None = None, return_caption: bool = True
    ) -> None:
        with open(tsv_path) as f:
            lines = f.readlines()

        assert lines[0].strip('\n') == 'filepath\ttitle'
        self.samples = [line.strip('\n').split('\t') for line in lines[1:]]

        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.return_caption = return_caption

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, caption = self.samples[index]
        img = self.img_transform(Image.open(path).convert('RGB'))

        if not self.return_caption:
            return img

        if self.txt_transform:
            caption = self.txt_transform(caption)

        return img, caption


class CombinedNet(Dataset):
    def __init__(
        self,
        index_path: str,
        in_class_index_path: str,
        class_mapping_path: str,
        transform: Callable,
        target_transform: Callable | None = None,
    ) -> None:
        with open(in_class_index_path) as f:
            in_class_index = json.load(f)
        self.wnid_to_idx = {wnid: int(label) for label, (wnid, _) in in_class_index.items()}

        with open(class_mapping_path) as f:
            class_mapping = json.load(f)
        self.cls_to_idx = {cls_name: i for i, cls_name in enumerate(class_mapping)}

        # sanity check
        assert self.cls_to_idx['banana'] == 13
        assert self.cls_to_idx['candle'] == 58
        assert self.cls_to_idx['lion'] == 174

        # create image net label to domain label mapping
        self.in_to_dn_idx = {
            in_idx: self.cls_to_idx[dn_cls_name]
            for dn_cls_name, in_indices in class_mapping.items()
            if in_indices is not None
            for in_idx in in_indices
        }

        with open(index_path) as f:
            lines = f.readlines()
        assert lines[0] == 'filepath\ttitle\n'
        paths = [line.strip('\n').split('\t')[0] for line in lines[1:]]
        self.samples = [(path, self._label_from_path(path)) for path in paths]

        self.transform = transform
        self.target_transform = target_transform

    def _label_from_path(self, path: str) -> int:
        identifier = path.split('/')[-2].replace('_', ' ').lower()
        if identifier in self.wnid_to_idx:
            assert identifier not in self.cls_to_idx
            in_label = self.wnid_to_idx[identifier]
            return self.in_to_dn_idx[in_label] + 1000 if in_label in self.in_to_dn_idx else in_label
        else:
            assert identifier in self.cls_to_idx
            return self.cls_to_idx[identifier] + 1000

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        img = self.transform(Image.open(path).convert('RGB'))

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class CompositionDataset(Dataset):
    # joint data loader to mit-states and ut-zappos

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
        target_transform: Callable | None = None,
        antonym_prompts: bool = False,
        also_return_obj_label: bool = False,
    ) -> None:
        self.root = root
        self.split = split

        # Load metadata
        self.metadata = torch.load(os.path.join(root, 'metadata_compositional-split-natural.t7'))

        # Load attribute-noun pairs for each split
        all_info, split_info = self.parse_split()
        self.attrs, self.objs, self.pairs = all_info
        self.train_pairs, self.valid_pairs, self.test_pairs = split_info

        # Get obj/attr/pair to indices mappings
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.idx2obj = {idx: obj for obj, idx in self.obj2idx.items()}
        self.idx2attr = {idx: attr for attr, idx in self.attr2idx.items()}
        self.idx2pair = {idx: pair for pair, idx in self.pair2idx.items()}
        self.unique_objs = list(set([noun for _, noun in self.pairs]))
        self.unique_attrs = list(set([attr for attr, _ in self.pairs]))
        self.antonym_data = load_antonym_data(root)

        assert (antonym_prompts and len(self.antonym_data) > 0) or not antonym_prompts

        # Get all data
        self.train_data, self.valid_data, self.test_data = self.get_split_info()
        if self.split == 'train':
            self.data = self.train_data
        elif self.split == 'valid':
            self.data = self.valid_data
        else:
            self.data = self.test_data

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        self.transform = transform
        self.target_transform = target_transform
        self.antonym_prompts = antonym_prompts
        self.also_return_obj_label = also_return_obj_label

    def parse_split(self):
        def parse_pairs(pair_path):
            with open(pair_path, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            os.path.join(self.root, 'compositional-split-natural', 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            os.path.join(self.root, 'compositional-split-natural', 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            os.path.join(self.root, 'compositional-split-natural', 'test_pairs.txt')
        )

        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return (all_attrs, all_objs, all_pairs), (tr_pairs, vl_pairs, ts_pairs)

    def get_split_info(self) -> tuple[list, list, list]:
        train_data, val_data, test_data = [], [], []
        for instance in self.metadata:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']
            image = image.split('/')[1]  # Get the image name without (attr, obj) folder
            image = os.path.join(self.root, 'images', ' '.join([attr, obj]), image)

            if (attr == 'NA') or ((attr, obj) not in self.pairs) or (settype == 'NA'):
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = {
                'image_path': image,
                'attr': attr,
                'obj': obj,
                'pair': (attr, obj),
                'attr_id': self.attr2idx[attr],
                'obj_id': self.obj2idx[obj],
                'pair_id': self.pair2idx[(attr, obj)],
            }
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        index = self.sample_indices[index]
        data_dict = self.data[index]

        img = self.transform(Image.open(os.path.join(data_dict['image_path'])))

        if self.target_transform:
            if self.antonym_prompts:
                captions = self.target_transform(
                    data_dict['pair'], self.antonym_data[data_dict['attr']], self.unique_objs
                )
            else:
                captions = self.target_transform(data_dict['pair'], self.unique_attrs, self.unique_objs)
            return img, (captions, self.attr2idx[data_dict['pair'][0]])
        if self.also_return_obj_label:
            return img, self.attr2idx[data_dict['pair'][0]], data_dict['obj_id']
        return img, self.attr2idx[data_dict['pair'][0]]


def load_antonym_data(data_root: str) -> dict:
    antonym_dict = {}
    antonym_path = os.path.join(data_root, 'adj_ants.csv')
    if not os.path.isfile(antonym_path):
        return antonym_dict
    with open(antonym_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip(',\n').split(',')
            antonym_dict[words[0]] = words[1:] if len(words) > 1 else []
    return antonym_dict
