import json
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass
class Aspect:
    abstract_theme: str
    course_of_action: str
    outcomes : str

@dataclass
class AspectTriplet:
    triplet_id: str
    anchor : str
    positive: str
    negative: str
    anchor_aspects: Aspect
    positive_aspects: Aspect
    negative_aspects: Aspect

    def pretty_print(self):
        print(f"Triplet ID: {self.triplet_id}\n")
        print(f"Anchor: {self.anchor}")
        print(f"  - Abstract Theme: {self.anchor_aspects.abstract_theme}")
        print(f"  - Course of Action: {self.anchor_aspects.course_of_action}")
        print(f"  - Outcomes: {self.anchor_aspects.outcomes}\n")
        print(f"Positive: {self.positive}")
        print(f"  - Abstract Theme: {self.positive_aspects.abstract_theme}")
        print(f"  - Course of Action: {self.positive_aspects.course_of_action}")
        print(f"  - Outcomes: {self.positive_aspects.outcomes}\n")
        print(f"Negative: {self.negative}")
        print(f"  - Abstract Theme: {self.negative_aspects.abstract_theme}")
        print(f"  - Course of Action: {self.negative_aspects.course_of_action}")
        print(f"  - Outcomes: {self.negative_aspects.outcomes}\n")


@dataclass
class EvalTriplet:
    triplet_id: str
    anchor : str
    positive: str
    negative: str



class AspectTripletDatasetTrain(Dataset):
    def __init__(self, aspect_triplets):
        self.aspect_triplets = aspect_triplets

    def __len__(self):
        return len(self.aspect_triplets)

    def __getitem__(self, idx):

        return {
            'anchor': self.aspect_triplets[idx].anchor,
            'positive': self.aspect_triplets[idx].positive,
            'negative': self.aspect_triplets[idx].negative,
            # .._theme : abstract_theme
            'anchor_theme': self.aspect_triplets[idx].anchor_aspects.abstract_theme,
            'positive_theme': self.aspect_triplets[idx].positive_aspects.abstract_theme,
            'negative_theme': self.aspect_triplets[idx].negative_aspects.abstract_theme,
            # .._action : course_of_action
            'anchor_action': self.aspect_triplets[idx].anchor_aspects.course_of_action,
            'positive_action': self.aspect_triplets[idx].positive_aspects.course_of_action,
            'negative_action': self.aspect_triplets[idx].negative_aspects.course_of_action,
            # .._outcome : outcomes
            'anchor_outcome': self.aspect_triplets[idx].anchor_aspects.outcomes,
            'positive_outcome': self.aspect_triplets[idx].positive_aspects.outcomes,
            'negative_outcome': self.aspect_triplets[idx].negative_aspects.outcomes,
        }
    
class AspectTripletDatasetDev(Dataset):
    def __init__(self, eval_data):
        self.eval_data = eval_data

    def __len__(self):
        return len(self.eval_data)
    
    def __getitem__(self, idx):
        return {
            'anchor': self.eval_data[idx].anchor,
            'positive': self.eval_data[idx].positive,
            'negative': self.eval_data[idx].negative,
        }

def save_aspect_triplets(triplets, filepath):
    with open(filepath, 'w') as f:
        for triplet in triplets:
            try:
                json_line = {
                    'triplet_id': triplet.triplet_id,
                    'anchor': triplet.anchor,
                    'positive': triplet.positive,
                    'negative': triplet.negative,
                    # theme
                    'anchor_theme': triplet.anchor_aspects.abstract_theme,
                    'positive_theme': triplet.positive_aspects.abstract_theme,
                    'negative_theme': triplet.negative_aspects.abstract_theme,
                    # course_of_action
                    'anchor_action' : triplet.anchor_aspects.course_of_action,
                    'positive_action' : triplet.positive_aspects.course_of_action,
                    'negative_action' : triplet.negative_aspects.course_of_action,
                    # outcomes
                    'anchor_outcome' : triplet.anchor_aspects.outcomes,
                    'positive_outcome' : triplet.positive_aspects.outcomes,
                    'negative_outcome' : triplet.negative_aspects.outcomes 
                }
                f.write(json.dumps(json_line) + '\n')
            except Exception as e:
                continue

def save_eval(eval_data, filepath):
    with open(filepath, 'w') as f:
        for triplet in eval_data:
            json_line = {
                'triplet_id': triplet.triplet_id,
                'anchor': triplet.anchor,
                'positive': triplet.positive,
                'negative': triplet.negative
            }
            f.write(json.dumps(json_line) + '\n')


def load_aspect_triplets(filepath):
    triplets = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            triplet = AspectTriplet(
                triplet_id=data['triplet_id'],
                anchor=data['anchor'],
                positive=data['positive'],
                negative=data['negative'],
                anchor_aspects=Aspect(
                    abstract_theme=data['anchor_theme'],
                    course_of_action=data['anchor_action'],
                    outcomes=data['anchor_outcome']
                ),
                positive_aspects=Aspect(
                    abstract_theme=data['positive_theme'],
                    course_of_action=data['positive_action'],
                    outcomes=data['positive_outcome']
                ),
                negative_aspects=Aspect(
                    abstract_theme=data['negative_theme'],
                    course_of_action=data['negative_action'],
                    outcomes=data['negative_outcome']
                )
            )
            triplets.append(triplet)
    return triplets

def load_eval_triplets(filepath):
    eval_triplets = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            triplet = EvalTriplet(
                triplet_id=data['triplet_id'],
                anchor=data['anchor'],
                positive=data['positive'],
                negative=data['negative']
            )
            eval_triplets.append(triplet)
    return eval_triplets



if __name__ == "__main__":
    # Example usage
    training_data = load_aspect_triplets("../data/processed/train_from_dev_w_aspect_aug_triplets.jsonl")
    dataset = AspectTripletDatasetTrain(training_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    eval_data = load_eval_triplets("../data/processed/eval_from_dev_triplets.jsonl")
    eval_dataset = AspectTripletDatasetDev(eval_data)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    # [TODO] Training loop 
    ...