from typing import List, Literal, Tuple
import itertools
import random
from collections import defaultdict
from PIL import Image
from pprint import pprint

VERTICAL_POSITIONING = {'Logo': [1], 'CTA Button': [1, 2, 3], 'Icon': [1, 2, 3], 'Product Image': [2],
               'Text Elements': [1,3], 'Infographic': [2], 'Banner': [1], 'Illustration': [2], 'Photograph': [2],
               'Mascot': [2], 'Testimonial Quotes': [2], 'Social Proof': [2, 1, 3], 'Seal or Badge': [3, 1, 2],
               'Graphs and Charts': [2], 'Decorative Elements': [3], 'Interactive Elements': [2],
               'Animation': [2], 'Coupon or Offer Code': [3], 'Legal Disclaimers or Terms': [3],
               'Contact Information': [3, 1, 2], 'Map or Location Image': [3], 'QR Code': [3, 1, 2]}

HORIZONTAL_POSITIONING = {'Logo': [1], 'CTA Button': [2, 1, 3], 'Icon': [1], 'Product Image': [1],
                          'Text Elements': [1], 'Infographic': [1], 'Banner': [2], 'Illustration': [2],
                          'Photograph': [2], 'Mascot': [1], 'Testimonial Quotes': [2], 'Social Proof': [3, 1, 2],
                          'Seal or Badge': [3, 1, 2], 'Graphs and Charts': [1], 'Decorative Elements': [3],
                          'Interactive Elements': [2], 'Animation': [2], 'Coupon or Offer Code': [3],
                          'Legal Disclaimers or Terms': [3], 'Contact Information': [3, 1, 2],
                          'Map or Location Image': [3], 'QR Code': [3, 1, 2]}

class ImageComposer:
    categories = Literal[
        "Background", "Logo", "CTA Button", "Icon", "Product Image", "Text Elements", "Infographic", "Banner", 
        "Illustration", "Photograph", "Mascot", "Testimonial Quotes", "Social Proof", "Seal or Badge", 
        "Graphs and Charts", "Decorative Elements", "Interactive Elements", "Animation", "Coupon or Offer Code", 
        "Legal Disclaimers or Terms", "Contact Information", "Map or Location Image", "QR Code"
    ]
    PositionSegment = Tuple[float, float]
    AlignmentPosition = Tuple[int, int]
    AlignmentPositions = List[AlignmentPosition]
    frame_images = List[Tuple[str, str, str]]

    def __init__(self, width: int, height: int, frames: List[frame_images]) -> None:
        self.width = width
        self.height = height
        self.frames = frames
        self.segments = ImageComposer.get_image_position_segments(width, height)
        self.generated_frames = []

    def generate_frames(self):
        self.compose_frames()
        return self.generated_frames

    def compose_frames(self) -> None:
        self.generated_frames = []

        for frame in self.frames:
            placement_items = []
            background_items = []

            for item in frame:
                if item[0].lower() == "background":  # Ensure background check is case-insensitive
                    background_items.append(item)
                else:
                    # Check if the item has exactly three elements
                    if len(item) != 3:
                        raise ValueError(f"Item {item} does not have exactly three elements")
                    placement_items.append(item)
            
            if not background_items:
                raise ValueError("No background found in frame")

            possibilities = ImageComposer.compute_positions([item[0] for item in placement_items])
            identified_locations = ImageComposer.select_diverse_positions(possibilities)
            adjusted_positions = self.calculate_adjusted_element_positions(identified_locations)
            
            # Debugging prints
            print("Frame:", frame)
            print("Placement Items:", placement_items)
            print("Identified Locations:", identified_locations)
            print("Adjusted Positions:", adjusted_positions)

            # Ensure category is included in placement_values
            placement_values = [(x[2], (y['x_start'], y['y_start']), (y['width'], y['height']), x[0]) for x, y in zip(placement_items, adjusted_positions)]
            
            for background in background_items:
                # Construct Frame for each background
                self.generated_frames.append(self.create_combined_image(background[2], placement_values))

    @staticmethod
    def compute_positions(elements: List[categories]) -> List[AlignmentPositions]:
        possible_positions = []

        # Iterate through each element to calculate its position combinations
        for element in elements:
            vertical_options = VERTICAL_POSITIONING[element]
            horizontal_options = HORIZONTAL_POSITIONING[element]
            combinations = list(itertools.product(vertical_options, horizontal_options))
            possible_positions.append(combinations)

        return possible_positions
    
    @staticmethod
    def select_diverse_positions(possible_positions: List[AlignmentPositions]) -> AlignmentPositions:
        position_frequency = defaultdict(int)

        def update_position_frequency(selected_position):
            position_frequency[selected_position] += 1

        selected_positions = []
        
        for positions in possible_positions:
            sorted_combinations = sorted(positions, key=lambda x: position_frequency[x])
            
            lowest_frequency = position_frequency[sorted_combinations[0]]
            lowest_freq_combinations = [pos for pos in sorted_combinations if position_frequency[pos] == lowest_frequency]
            
            selected_position = random.choice(lowest_freq_combinations)
            selected_positions.append(selected_position)
            
            update_position_frequency(selected_position)
        
        return selected_positions

    @staticmethod
    def get_image_position_segments(width: float, height: float, vm: float = 0.6, vo: float = 0.2, hm: float = 0.6, ho: float = 0.2) -> Tuple[List[PositionSegment], List[PositionSegment]]:
        """Divide Image based on percentage for vertical and horizontal segments."""
        
        if vm + vo * 2 > 1 or hm + ho * 2 > 1:
            raise ValueError("Sum of percentages exceeds 100% for either vertical or horizontal segments.")
        
        vertical_mid = height * vm
        vertical_outer = height * vo
        horizontal_mid = width * hm
        horizontal_outer = width * ho

        vertical_segments = [
            (0, vertical_outer),
            (vertical_outer, vertical_outer + vertical_mid),
            (vertical_outer + vertical_mid, height)
        ]
        
        horizontal_segments = [
            (0, horizontal_outer),
            (horizontal_outer, horizontal_outer + horizontal_mid),
            (horizontal_outer + horizontal_mid, width)
        ]

        return vertical_segments, horizontal_segments

    def calculate_adjusted_element_positions(self, positions: AlignmentPositions) -> List[dict]:
        adjusted_positions = []

        for pos in positions:
            vertical_segment = self.segments[0][pos[0] - 1]
            horizontal_segment = self.segments[1][pos[1] - 1]

            x_start = horizontal_segment[0]
            y_start = vertical_segment[0]
            width = horizontal_segment[1] - horizontal_segment[0]
            height = vertical_segment[1] - vertical_segment[0]

            adjusted_positions.append({
                'x_start': x_start,
                'y_start': y_start,
                'width': width,
                'height': height
            })

        return adjusted_positions

    def create_combined_image(self, background_path: str, elements_positions: list) -> Image.Image:
        """
        Create a combined image based on background and elements' positioning and sizing.
        
        :param background_path: Path to the background image.
        :param elements_positions: A list of tuples containing 'image_path', 'start_point', and 'dimensions'.
        """
        background_image = Image.open(background_path).convert("RGBA")
        background_image = background_image.resize((self.width, self.height))

        print(f"Creating combined image with background: {background_path}")
        print(f"Elements Positions: {elements_positions}")

        for element in elements_positions:
            if len(element) != 4:
                raise ValueError(f"Element {element} does not have exactly four components")

            element_path, (x_start, y_start), (width, height), category = element
            element_image = Image.open(element_path).convert("RGBA")
            element_image = self.resize_image(element_image, int(width), int(height))

            print(f"Placing element {element_path} at ({x_start}, {y_start}) with size ({width}, {height})")

            background_image.paste(element_image, (int(x_start), int(y_start)), element_image)

        return background_image

    @staticmethod
    def resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        aspect_ratio = image.width / image.height
        
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image





