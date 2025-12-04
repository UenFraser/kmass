# LIBRARIES NEEDED
import streamlit as st
import mesa
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib.pyplot as plt
import colorsys
from matplotlib import colors as mcolors
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import itertools
import heapq
from scipy.stats import ttest_ind, ks_2samp
import seaborn as sns

########################################### EVENT DRIVEN MODEL ########################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ---------------------------------------------------------
# EVENT STRUCTURE
# ---------------------------------------------------------
class Event:
    def __init__(self, time, event_type, agent=None, dish=None):
        self.time = time
        self.event_type = event_type
        self.agent = agent
        self.dish = dish

    def __lt__(self, other):
        return self.time < other.time
# ---------------------------------------------------------
# EVENT QUEUE MANAGER
# ---------------------------------------------------------
class EventQueue:
    def __init__(self):
        self.queue = []
        self.counter = itertools.count()  # tie breaker

    def push(self, event):
        heapq.heappush(self.queue, (event.time, next(self.counter), event))
    def pop(self):
        if self.queue:
            return heapq.heappop(self.queue)[2]
        return None
    def empty(self):
        return len(self.queue) == 0

# ---------------------------------------------------------
# BASE AGENT
# ---------------------------------------------------------
class Agent:
    def __init__(self, name, kitchen):
        self.name = name
        self.kitchen = kitchen
        self.state = "idle"
        self.specialization = None

        # for staff utilization metrics
        self.total_busy_time = 0      # cumulative busy time in seconds
        self.last_state_change = 0    # last timestamp when state changed

    def __repr__(self):
        return f"{self.name}({self.state})"
# ---------------------------------------------------------
# PREP / COOK / WAITER
# ---------------------------------------------------------
class PrepChef2(Agent):
    def __init__(self, unique_id, kitchen, specialization):
        super().__init__(f"PrepChef{unique_id}", kitchen)
        self.specialization = specialization
class CookChef2(Agent):
    def __init__(self, unique_id, kitchen, specialization):
        super().__init__(f"CookChef{unique_id}", kitchen)
        self.specialization = specialization
        self.current_load = 0
        self.max_load = 3
class Waiter2(Agent):
    def __init__(self, unique_id, kitchen, role):
        super().__init__(f"Waiter{unique_id}", kitchen)
        self.special = role

# ---------------------------------------------------------
# KITCHEN ENVIRONMENT (EVENT-DRIVEN)
# ---------------------------------------------------------
class ED_Kitchen:
    def __init__(self, menu, agentsVP=0, agentsMP=0, agentsPAR=1,
                       agentsG=0, agentsF=0, agentsS=0, agentsCAR=6,
                       agentsW=4, agentsDW=1, n_food=30):
        self.time = 0
        self.MAX_TIME = 4 * 60 * 60  # 4 hours in seconds
        self.event_queue = EventQueue()
        self.menu = menu
        self.n_food = n_food
        self.orders = []
        self.dishes = []
        self.total_stoves = 10
        self.used_stoves = 0
        self.finished_dishes = []
        # basic simulated outside work - mimic waiter busy at outside
        self.outsideWork = {
            "serve_customer": 90,          # 1.5 minutes — walk to table, deliver food, small talk
            "take_order": 120,             # 2 minutes — chat, write order, confirm details
            "bring_drinks": 60,            # 1 minute — fetch and deliver drinks
            "clean_table": 180,            # 3 minutes — wipe, organize, reset
            "deliver_bill": 45,            # 45 seconds — hand over bill and wait briefly
            "collect_payment": 75,         # 1.25 minutes — handle cash/card, return change
            "chat_with_customers": 90,     # 1.5 minutes — occasional small talk or feedback
            "return_to_kitchen": 30        # 30 seconds — walking back inside
        }
        self.dirty_plates = 0
        self.carry_plates = 0
        self.PLATE_THRESHOLD = 20
        self.dishwashing_queue = 0
        self.prep_filth = 0
        self.FILTH_LIMIT = 10
        self.rice_amount = 100
        self.RICE_BATCH_SIZE = 100

        # ----------------- SIMPLE ORDER SCRIPT -----------------
        self.orders = generate_simple_orders(menu=self.menu, total_orders=self.n_food, window_sec=self.MAX_TIME)

        # schedule order arrivals
        for order in self.orders:
            for dish in order["dishes"]:
                self.event_queue.push(Event(order["arrival_time"], "order_arrival", dish=dish))
                self.dishes.append(dish)

        ########################## SPAWNING AGENTS ########################
        # Agent lists
        self.prep_chefs = []
        self.cook_chefs = []
        self.waiters = []
        # IDs
        P_id = 1
        C_id = 1
        W_id = 1
        # --- Prep Chefs ---
        for i in range(agentsVP):
            chef = PrepChef2(P_id, self, "Veggie_Prep")
            self.prep_chefs.append(chef)
            P_id += 1
        for i in range(agentsMP):
            chef = PrepChef2(P_id, self, "Meat_Prep")
            self.prep_chefs.append(chef)
            P_id += 1
        for i in range(agentsPAR):
            chef = PrepChef2(P_id, self, "All_Rounder")
            self.prep_chefs.append(chef)
            P_id += 1

        # --- Cook Chefs --- (times 3 to mimic cook's ability to multi-task up to 3 dishes max only)
        for i in range(agentsG * 3):
            chef = CookChef2(C_id, self, "Grill")
            self.cook_chefs.append(chef)
            C_id += 1
        for i in range(agentsF * 3):
            chef = CookChef2(C_id, self, "Fry")
            self.cook_chefs.append(chef)
            C_id += 1
        for i in range(agentsS * 3):
            chef = CookChef2(C_id, self, "Stew")
            self.cook_chefs.append(chef)
            C_id += 1
        for i in range(agentsCAR * 3):
            chef = CookChef2(C_id, self, "All_Rounder")
            self.cook_chefs.append(chef)
            C_id += 1

        # --- Waiters ---
        for i in range(agentsW):
            w = Waiter2(W_id, self, "Server")
            self.waiters.append(w)
            W_id += 1
        for i in range(agentsDW):
            dw = Waiter2(W_id, self, "Dishwasher")
            self.waiters.append(dw)
            W_id += 1

        # Queues for dishes
        self.prep_queue = []
        self.cook_queue = []
        self.serve_queue = []

    # -----------------------------------------------------
    # STAFF UTILIZATION RECORD
    # -----------------------------------------------------
    def update_agent_busy_time(self, agent, new_time):
        busy_states = ["busy", "outside_work", "cleaning", "collecting_plates", "cooking_rice"]
        
        # accumulate busy time if agent was busy
        if agent.state in busy_states:
            agent.total_busy_time += new_time - agent.last_state_change

        # update the timestamp of the last state change
        agent.last_state_change = new_time

    # -----------------------------------------------------
    # EVENT SCHEDULING
    # -----------------------------------------------------
    def schedule(self, time, event_type, agent=None, dish=None):
        event = Event(time, event_type, agent, dish)
        self.event_queue.push(event)

    # -----------------------------------------------------
    # EVENT HANDLER
    # -----------------------------------------------------
    def process_event(self, event):
        # # # # # # # # # # # # # # NORMAL SEQUENCE # # #  #  # # # # # # # # # Z#
        if event.event_type == "order_arrival":
            dish = event.dish
            dish.arrival_time = event.time
            self.prep_queue.append(dish)
            self.try_assign_prep()

        elif event.event_type == "prep_start":
            chef = event.agent
            dish = event.dish
            self.update_agent_busy_time(chef, event.time)
            chef.state = "busy"
            dish.prep_start = event.time
            self.schedule(event.time + dish.prepare, "prep_end", chef, dish)

        elif event.event_type == "prep_end":
            chef = event.agent
            dish = event.dish
            self.update_agent_busy_time(chef, event.time)
            chef.state = "idle"
            dish.prep_end = event.time

            self.cook_queue.append(dish)
            dish.cook_queue_enter = event.time # cook waiting time start
            self.try_assign_cook()

            self.prep_filth += 1
            # Cleaning required
            if self.prep_filth >= self.FILTH_LIMIT:
                self.schedule(self.time, "clean_start", chef)
                return

            self.try_assign_prep()
            # print("PREP DONE:", dish) # DEBUGGING

        elif event.event_type == "cook_start":
            chef = event.agent
            dish = event.dish
            self.update_agent_busy_time(chef, event.time)
            chef.state = "busy"
            dish.cook_wait_time = event.time - dish.cook_queue_enter # cook waiting time end
            dish.cook_start = event.time
            self.schedule(event.time + dish.cook, "cook_end", chef, dish)

        elif event.event_type == "cook_end":
            chef = event.agent
            dish = event.dish

            # record cook end time
            dish.cook_end = event.time
            self.update_agent_busy_time(chef, event.time)
            chef.state = "idle"

            # use rice
            self.rice_amount -= dish.rice
            if self.rice_amount < 0:
                self.rice_amount = 0

            # move dish to serve queue
            self.serve_queue.append(dish)
            dish.serve_queue_enter = event.time # serve waiting time start
            self.try_assign_waiter()

            if self.rice_amount <= 0:
                self.schedule(self.time, "cook_rice_start", chef)
                return

            self.try_assign_cook()
            # print("COOK DONE:", dish) # DEBUGGING

        elif event.event_type == "serve_start":
            waiter = event.agent
            dish = event.dish
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "busy"
            dish.serve_wait_time = event.time - dish.serve_queue_enter # serve waiting time end
            dish.serve_start = event.time
            self.schedule(event.time + dish.serve, "serve_end", waiter, dish)
            # print("SERVE START:", dish) # DEBUGGING

        elif event.event_type == "serve_end":
            waiter = event.agent
            dish = event.dish
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "idle"
            dish.serve_end = event.time

            # Store the completed dish for metrics later
            self.finished_dishes.append(dish)
            
            # Every served dish increases dirty plates
            self.dirty_plates += 1

            # Trigger outside errands randomly (simulated distraction)
            if random.random() < 0.15 and waiter.special == "Server":
                num_tasks = random.choices([1,2,3], weights=[0.6,0.3,0.1])[0]
                chosen_tasks = random.sample(list(self.outsideWork.values()), num_tasks)
                total_duration = sum(chosen_tasks)
                
                self.schedule(self.time, "outside_start", agent=waiter)
                self.schedule(self.time + total_duration, "outside_end", agent=waiter)
                return  # waiter is busy, skip regular assign

            # Otherwise regular behavior
            self.try_assign_waiter()
            # print("SERVE DONE:", dish) # DEBUGGING

        # ---------------- OUTSIDE WORK ----------------
        elif event.event_type == "outside_start":
            waiter = event.agent
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "outside_work"
            # nothing else needed here; the end event is already scheduled
        elif event.event_type == "outside_end":
            waiter = event.agent
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "idle"
            # after finishing outside work, try to assign waiter to tasks again
            self.try_assign_waiter()

        # ---------------- COLLECT PLATES ----------------
        elif event.event_type == "collect_plates_start":
            waiter = event.agent
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "collecting_plates"
            
            # schedule the end of plate collection, assume each plate takes 10 seconds
            collection_duration = self.dirty_plates * 10
            self.carry_plates = self.dirty_plates
            self.dirty_plates = 0  # reset dirty plates after scheduling collection
            self.schedule(self.time + collection_duration, "collect_plates_end", agent=waiter)
        elif event.event_type == "collect_plates_end":
            waiter = event.agent
            self.update_agent_busy_time(waiter, event.time)
            waiter.state = "idle"

            # Create a synthetic dishwasher job
            self.dishwashing_queue += self.carry_plates
            self.carry_plates = 0
            
            # after collecting plates, try to assign waiter to tasks again
            self.try_assign_waiter()

        # ---------------- WASHING PLATES ----------------
        elif event.event_type == "wash_start":
            dishwasher = event.agent
            self.update_agent_busy_time(dishwasher, event.time)
            dishwasher.state = "busy"

            # take up to a batch of plates (or all if less)
            batch_size = min(self.dishwashing_queue, 20)  # for example, 20 plates per DW
            self.dishwashing_queue -= batch_size  # reduce queue by batch taken

            # schedule end of washing
            wash_duration = batch_size * 10  # 10 seconds per plate, adjust as needed
            self.schedule(self.time + wash_duration, "wash_end", dishwasher)
        elif event.event_type == "wash_end":
            dishwasher = event.agent
            self.update_agent_busy_time(dishwasher, event.time)
            dishwasher.state = "idle"

            # after finishing, check if more work is needed or assign more dishwashers if plates remain
            self.try_assign_waiter()

        # ---------------- CLEANING PREP CHEF ----------------
        elif event.event_type == "clean_start":
            chef = event.agent
            self.update_agent_busy_time(chef, event.time)
            chef.state = "cleaning"
            
            # schedule end of cleaning (can be tuned)
            CLEAN_TIME = 60
            self.schedule(self.time + CLEAN_TIME, "clean_end", chef)
        elif event.event_type == "clean_end":
            chef = event.agent
            self.update_agent_busy_time(chef, event.time)
            chef.state = "idle"
            
            # reset filth counter after cleaning
            self.prep_filth = 0
            
            # after cleaning, try assigning prep tasks again
            self.try_assign_prep()

        # ---------------- COOKING RICE CASE ----------------
        elif event.event_type == "cook_rice_start":
            chef = event.agent
            self.update_agent_busy_time(chef, event.time)
            chef.state = "cooking_rice"

            RICE_COOK_TIME = 300  # 5 minutes, adjust as needed
            self.schedule(self.time + RICE_COOK_TIME, "cook_rice_end", chef)
        elif event.event_type == "cook_rice_end":
            chef = event.agent
            self.update_agent_busy_time(chef, event.time)
            chef.state = "idle"
            
            # refill rice for next dishes
            self.rice_amount = self.RICE_BATCH_SIZE  # 100
            self.used_stoves -= 1
            
            # after rice is ready, try assigning cook tasks again
            self.try_assign_cook()

    # -----------------------------------------------------
    # ASSIGNMENT FUNCTIONS
    # -----------------------------------------------------
    def try_assign_prep(self):
        """
        Assign dishes in prep_queue to idle prep chefs
        with matching specialization.
        """
        for chef in self.prep_chefs:
            if chef.state != "idle":
                continue  # skip busy chefs

            # look for a dish that matches chef specialization
            for i, dish in enumerate(self.prep_queue):
                if chef.specialization == "All_Rounder" or chef.specialization == dish.preptype:
                    # assign this dish
                    self.schedule(self.time, "prep_start", chef, dish)
                    self.prep_queue.pop(i)
                    break  # assign only one dish at a time

    def try_assign_cook(self):
        """
        Assign dishes in cook_queue to idle cook chefs
        respecting specialization.
        """
        for chef in self.cook_chefs:
            # only assign if chef is idle
            if chef.state != "idle":
                continue
            # find a matching dish
            for i, dish in enumerate(self.cook_queue):

                if chef.specialization == "All_Rounder" or chef.specialization == dish.cooktype:
                    # assign the dish
                    self.schedule(self.time, "cook_start", chef, dish)
                    self.cook_queue.pop(i)
                    chef.state = "busy"

    def try_assign_waiter(self):
        """
        Assign dishes in serve_queue to idle waiters.
        Handles servers and dishwashers safely.
        """
        for waiter in self.waiters:
            # Skip if busy
            if waiter.state != "idle":
                continue

            # Dishwashers: handle washing first
            if getattr(waiter, "special", None) == "Dishwasher" and self.dishwashing_queue > 0:
                waiter.state = "busy"
                self.schedule(self.time, "wash_start", waiter)
                continue

            # Collect dirty plates if threshold reached
            if self.dirty_plates >= self.PLATE_THRESHOLD:
                waiter.state = "busy"
                self.schedule(self.time, "collect_plates_start", waiter)
                continue

            # Servers: assign dishes
            if getattr(waiter, "special", None) == "Server" and len(self.serve_queue) > 0:
                dish = self.serve_queue.pop(0)  # get first dish
                waiter.state = "busy"
                self.schedule(self.time, "serve_start", waiter, dish)

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    def run(self):
        #print("Simulation starting...")  # <---- DEBUGGING
        while not self.event_queue.empty():
            event = self.event_queue.pop()

            # STOPPING CONDITION: 4 hours max
            if event.time > self.MAX_TIME:
                self.time = self.MAX_TIME
                break

            self.time = event.time
            self.process_event(event)
        #print("Simulation finished")  # <---- DEBUGGING


########################################## TICK-BASED MODEL ###########################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------------------------------
#                                       Agent Classes
# -----------------------------------------------------------------------------------------------------
################################### ## Chef Group Agents ############################################
class CookChef(mesa.Agent):
    '''
    A Chef agent that cooks the food based on food type.
    Deducts the amount of the Dish cooking steps.
    '''

    def __init__(self, unique_id, model, special, work=0, x=None, y=None, paths=None, spawn_cell=None):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.spawn_cell = spawn_cell
        self.paths = paths  # dictionary of paths for this spawn cell

        # multi-tasking works
        self.work = work
        self.work2 = work
        self.work3 = work

        self.wait = 0

        self.special = special
        self.current_task = None
        self.dish_load = [] # allowing agent to multi-task dishes
        self.path = [] 
        self.state = "idle"

        # agent metric records
        self.work_durations = []  # stores how long they worked each time
        self.idle_durations = []  # stores idle gaps
        self.sideTrack_durations = [] # durations of other non-productive but active tasks
        self.last_busy_end = 0    # used to track idle time

    # helper to reset agent state
    def reset_agent(self):
        #self.work = 0
        # self.current_task = None
        self.path = []
        self.state = "idle"

    def follow_path(self, path_keys):
        """Start following one or more paths, with optional reversing."""
        full_path = []

        # Make sure path_keys is always a list
        if isinstance(path_keys, (str, tuple)):
            path_keys = [path_keys]

        for key in path_keys:
            reverse = False
            if isinstance(key, tuple):
                path_key, reverse = key
            else:
                path_key = key

            if path_key in self.paths:
                path_options = self.paths[path_key]

                # If it's a single path (list of coords)
                if isinstance(path_options[0][0], int):
                    path = list(path_options)
                else:
                    # Multiple alternative paths → choose one
                    path = list(random.choice(path_options))

                if reverse:
                    path.reverse()

                full_path.extend(path)

        self.path = full_path

    def step_along(self):
        """Move one step forward if there is a path."""
        if self.path:
            self.x, self.y = self.path.pop(0)  # consume from the *copy*

    def cooking_move(self):
        """Cook's Behavior."""
        
        if self.state == "idle":  # assign work
            # Check if any dish is almost done (finish within 10 time units)
            if self.dish_load:
                current_time = self.model.time
                for dish in self.dish_load:
                    end_time = dish[1]
                    if 0 <= end_time - current_time <= 10:
                        if (self.x, self.y) == self.spawn_cell: 
                            self.state = "working"
                        else:
                            self.follow_path(("to_pass", True))
                            self.state = "return_stove"
                        return

            # special rice case
            if self.model.rice == 0 and self.special == "all-rounder" and len(self.dish_load) < 4:
                self.current_task = "rice"
                
                if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                    self.follow_path(("cook_rice", True))
                else:
                    self.follow_path("grab_rice")
                self.state = "to_rice_station"

            # normal dish cooking
            elif self.model.cooking_queue and len(self.dish_load) < 4:
                for order in list(self.model.cooking_queue):
                    if self.special == order.cooktype or self.special == "all-rounder":
                        if order in self.model.cooking_queue:
                            self.current_task = order
                            self.model.cooking_queue.remove(order)
                        
                            if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                                self.follow_path("to_island")
                            else:
                                self.follow_path("go_back") # when agent is at the pass table
                            self.state = "to_island_counter"
                            break

            elif self.model.prepTable:
                for order in list(self.model.prepTable):
                    if self.special == order.cooktype or self.special == "all-rounder":
                        if order in self.model.prepTable:
                            self.current_task = order
                            self.model.prepTable.remove(order)

                            if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                                self.follow_path("to_prepA")
                            else:
                                self.follow_path("to_prepB") # when agent is at the pass table
                            self.state = "to_prep_table"
                            break
                else: # when agent can't find match
                    if self.model.prepTable:
                        self.current_task = self.model.prepTable.popleft()
                        if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                            self.follow_path("to_prepA")
                        else:
                            self.follow_path("to_prepB") # when agent is at the pass table
                        self.state = "to_prep_table"

            else:  # nothing to cook
                if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                    self.state = "idle"
                else:
                    self.follow_path(("to_pass", True))
                    self.state = "go_back"

        elif self.state == "to_island_counter":
            self.step_along()
            if not self.path:  # arrived
                self.follow_path("go_cook")
                self.state = "to_stove"

        elif self.state == "to_prep_table":
            self.step_along()
            if not self.path:  # arrived
                if self.special == self.current_task.cooktype or self.special == "all-rounder":
                    self.follow_path(("to_prepA", True)) # reversed path to return
                    self.state = "to_stove"
                else:
                    # specialization mismatch → put dish at the island counter
                    self.follow_path("prep_to_island")
                    self.state = "send_to_island"

        elif self.state == "send_to_island":
            self.step_along()
            if not self.path:  # arrived
                self.model.cooking_queue.append(self.current_task)
                self.reset_agent()

                if self.model.prepTable:
                    for order in list(self.model.prepTable):
                        if self.special == order.cooktype or self.special == "all-rounder":
                            if order in self.model.prepTable:
                                self.current_task = order
                                self.model.prepTable.remove(order)
                                self.follow_path(("prep_to_island", True))
                                self.state = "to_prep_table"
                                break
                    else: # when agent can't find a match :(
                        if self.model.prepTable:
                            self.current_task = self.model.prepTable.popleft()
                            self.follow_path(("prep_to_island", True))
                            self.state = "to_prep_table"
                elif self.model.cooking_queue:
                    for order in list(self.model.cooking_queue):
                        if self.special == order.cooktype or self.special == "all-rounder":
                            if order in self.model.cooking_queue:
                                self.current_task = order
                                self.model.cooking_queue.remove(order)
                                self.follow_path("go_cook")
                                self.state = "to_stove"
                                break
                    else: # when agent can't find a match :(
                        self.follow_path("go_cook") # same path to return, not actually cooking
                        self.state = "go_back"
                else:
                    self.follow_path("go_cook") # same path to return, not actually cooking
                    self.state = "go_back"

        elif self.state == "go_back" or self.state == "to_pass_table":
            self.step_along()
            if not self.path:  # arrived
                self.reset_agent()

        elif self.state == "return_stove":
            self.step_along()
            if not self.path:  # arrived
                self.state = "working"

        elif self.state == "to_rice_station":
            self.step_along()
            if not self.path:  # arrived
                self.follow_path("cook_rice")
                self.state = "to_stove"

        elif self.state == "to_stove":
            self.step_along()
            if not self.path:  # arrived
                # set start time
                if self.current_task != "rice":
                    self.current_task.cook_start = self.model.time
                    duration = self.current_task.cook
                else:
                    duration = 5  # rice fixed time

                # compute end time
                end_time = self.model.time + duration

                # assign to free slot
                if self.work == 0:
                    self.work = end_time
                    self.dish_load.append([self.current_task, end_time, 1])
                elif self.work2 == 0:
                    self.work2 = end_time
                    self.dish_load.append([self.current_task, end_time, 2])
                else:
                    self.work3 = end_time
                    self.dish_load.append([self.current_task, end_time, 3])

                self.state = "working"

        elif self.state == "working":
            finished_dishes = []
            current_time = self.model.time

            # Check if any dish has reached its end time
            # dish = [task_or_rice, end_time, slot]
            for dish in self.dish_load:
                end_time = dish[1]

                if current_time >= end_time:
                    # Free the slot
                    if dish[2] == 1:
                        self.work = 0
                    elif dish[2] == 2:
                        self.work2 = 0
                    else:
                        self.work3 = 0

                    # Rice case
                    if dish[0] == "rice":
                        self.model.rice += 50
                    else:
                        dish[0].cook_end = current_time

                    finished_dishes.append(dish)

            # If something finished
            if finished_dishes:
                for dish in finished_dishes:
                    self.dish_load.remove(dish)

                    if dish[0] != "rice":
                        self.model.serving_queue.append(dish[0])
                        self.model.trash += random.randint(1, 2)

                # If any normal dishes finished → go to pass table
                if any(d[0] != "rice" for d in finished_dishes):
                    self.follow_path("to_pass")
                    self.state = "to_pass_table"

                # Reset wait only if NOTHING is cooking
                if not self.dish_load:
                    self.wait = 0

                return  # done handling finished dishes

            # Nothing finished → handle general waiting
            if self.wait < 120:
                self.wait += 1
            else:
                self.wait = 0
                self.state = "idle"

class PrepChef(mesa.Agent):
    """
    A Chef agent that prepares the ingredients based on food type. 
    Deducts the amount of the Dish prepping steps.
    """

    def __init__(self, unique_id, model, special, work=0, x=None, y=None, paths=None, spawn_cell=None):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.spawn_cell = spawn_cell
        self.paths = paths  # dictionary of paths for this spawn cell

        self.work = work
        self.special = special
        self.current_dish = None
        self.path = [] 
        self.state = "idle" 
        self.grabbing = False 

        # agent metric records
        self.work_durations = []  # stores how long they worked each time
        self.idle_durations = []  # stores idle gaps
        self.sideTrack_durations = [] # durations of other non-productive but active tasks
        self.last_busy_end = 0    # used to track idle time

    def reset_agent(self):
        """Reset the agent back to idle state and clear path/target."""
        self.work = 0
        self.current_dish = None
        self.path = []
        self.state = "idle"

    def follow_path(self, path_keys):
        """Start following one or more paths, with optional reversing."""
        full_path = []

        # Make sure path_keys is always a list
        if isinstance(path_keys, (str, tuple)):
            path_keys = [path_keys]

        for key in path_keys:
            reverse = False
            if isinstance(key, tuple):
                path_key, reverse = key
            else:
                path_key = key

            if path_key in self.paths:
                path_options = self.paths[path_key]

                # If it's a single path (list of coords)
                if isinstance(path_options[0][0], int):
                    path = list(path_options)
                else:
                    # Multiple alternative paths → choose one
                    path = list(random.choice(path_options))

                if reverse:
                    path.reverse()

                full_path.extend(path)

        self.path = full_path


    def step_along(self):
        """Move one step forward if there is a path."""
        if self.path:
            self.x, self.y = self.path.pop(0)  # consume from the *copy*

    def prepping_move(self):
        """Prep's Behavior."""

        if self.state == "idle":  # assign work
            # cleaning task
            if self.model.trash > 30 and self.special == "all-rounder" and not self.model.throwing_trash:
                self.model.throwing_trash = True   # claim cleaning task
                self.current_dish = "clean_kitchen"
                self.work = 5
                if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                    self.follow_path(("after_trash", True))
                else:  # not at spawn cell
                    self.follow_path("to_trash")
                self.state = "throw_trash"

            # prepping task
            elif self.model.recievedOrder:
                self.current_dish = self.model.recievedOrder.popleft()
                if self.special == self.current_dish.preptype or self.special == "all-rounder":
                    self.follow_path("grab_ingred")
                    self.state = "grab_ingred" # go grab ingredients from inventory
                    self.grabbing = True
                else:
                    # specialization doesn’t match → return dish
                    self.model.recievedOrder.appendleft(self.current_dish)
                    self.reset_agent()

            elif self.model.prepTable and not self.model.recievedOrder:
                self.current_dish = self.model.prepTable.popleft()
                self.follow_path("to_island")
                self.state = "to_island_counter"

            # nothing to do → remain idle
            else:
                self.state = "idle"

        elif self.state == "throw_trash" or self.state == "walk_back":
            self.step_along()
            if not self.path:  # arrived
                if self.state == "walk_back":
                    self.current_dish.prep_start = self.model.time
                self.state = "working"

        elif self.state == "go_back":
            self.step_along()
            if not self.path:  # arrived
                self.state = "idle"

        elif self.state == "grab_ingred":
            self.step_along()
            if not self.path:  # arrived
                self.work = 3
                self.state = "working"

        elif self.state == "to_island_counter":
            self.step_along()
            if not self.path:  # arrived
                self.model.cooking_queue.append(self.current_dish)
                self.model.trash += random.randint(1, 3)
                self.reset_agent()
                self.follow_path("go_back")
                self.state = "go_back"

        elif self.state == "working":
            if self.work > 1:
                self.work -= 1
            elif self.work == 1:  # work done
                self.work = 0
                if self.current_dish == "clean_kitchen":
                    self.model.throwing_trash = False   # release flag
                    self.model.trash = 0
                    self.reset_agent()
                    self.follow_path("after_trash")
                    self.state = "go_back"
                elif self.grabbing:
                    self.work = self.current_dish.prepare
                    self.grabbing = False
                    self.follow_path(("grab_ingred", True))
                    self.state = "walk_back"
                else:
                    self.current_dish.prep_end = self.model.time
                    self.model.prepTable.append(self.current_dish)
                    self.reset_agent()

#################################### Waiter Group Agents ############################################
class Waiter(mesa.Agent):
    """
    A Waiter agent that can serve dishes and clean used plates.
    """

    def __init__(self, unique_id, model, special, x=None, y=None, paths=None, spawn_cell=None):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.spawn_cell = spawn_cell
        self.paths = paths  # dictionary of paths for this spawn cell

        self.work = 0
        self.broughtPlates = 0
        self.special = special
        self.current_dish = None
        self.orderList = []
        self.path = []
        self.state = "idle"
        self.wait = 0

        # agent metric records
        self.work_durations = []  # stores how long they worked each time
        self.idle_durations = []  # stores idle gaps
        self.sideTrack_durations = [] # durations of other non-productive but active tasks
        self.last_busy_end = 0    # used to track idle time

    def reset_agent(self):
        """Reset agent state for new tasks."""
        self.work = 0
        self.current_dish = None
        self.path = []
        self.broughtPlates = 0
        self.orderList = []
        self.state = "idle"

    def follow_path(self, path_keys):
        """Start following one or more paths, with optional reversing."""
        full_path = []

        # Make sure path_keys is always a list
        if isinstance(path_keys, (str, tuple)):
            path_keys = [path_keys]

        for key in path_keys:
            reverse = False
            if isinstance(key, tuple):
                path_key, reverse = key
            else:
                path_key = key

            if path_key in self.paths:
                path_options = self.paths[path_key]

                # If it's a single path (list of coords)
                if isinstance(path_options[0][0], int):
                    path = list(path_options)
                else:
                    # Multiple alternative paths → choose one
                    path = list(random.choice(path_options))

                if reverse:
                    path.reverse()

                full_path.extend(path)

        self.path = full_path

    def step_along(self):
        """Move one step forward if there is a path."""
        if self.path:
            self.x, self.y = self.path.pop(0)  # consume from the *copy*


    def serving_move(self):
        """Waiter's Behavior."""

        if self.special == "server": # for server type waiters
            if self.state == "idle": # assign work

                if (self.x, self.y) == self.spawn_cell: # go outside immediately when spawned
                        self.follow_path("to_door")
                        self.state = "go_outside"

                # bring dirty plates in
                elif self.model.DirtyPlates >= 10 and self.model.takeback_plates < self.model.max_returnees and (self.x, self.y) in self.model.being_outside:
                    self.model.takeback_plates += 1   # claim slot
                    self.current_dish = "return_plates"
                    self.broughtPlates += self.model.DirtyPlates
                    self.follow_path("to_sink")
                    self.state = "to_sink"

                # help cleaning task
                elif self.model.cleaningPile >= 30 and self.model.cleaning_in_progress < self.model.max_cleaners and (self.x, self.y) in self.model.being_outside:
                    self.model.cleaning_in_progress += 1   # claim slot
                    self.current_dish = "clean_up"
                    self.work = 3
                    self.follow_path("to_sink")
                    self.state = "to_sink"

                # serving task, if there's something at the queue
                elif self.model.serving_queue:
                    if (self.x, self.y) == (7,6):
                        self.follow_path("after_order")
                        self.state = "to_pass"
                    else: # means the agent is at the door
                        self.follow_path("go_back") 
                        self.state = "going_back"

                # bring new order from the customer outside to the prep
                elif self.model.fromOutside and (self.x, self.y) in self.model.being_outside:
                    taken_order = self.model.fromOutside.popleft()
                    self.orderList.extend(taken_order)
                    self.follow_path("give_order") 
                    self.state = "give_order"

                else: # if nothing is at the queue, go back outside and wait
                    if (self.x, self.y) == (7,6):
                        self.follow_path(("give_order", True)) 
                        self.state = "go_outside"
                    else: # means the agent is at the door
                        self.state = "idle"

            elif self.state == "to_sink":
                self.step_along()
                if not self.path:  # arrived
                    if self.current_dish == "return_plates":
                        self.model.cleaningPile += self.model.DirtyPlates
                        self.reset_agent()
                        if self.model.serving_queue: # pick up dish if there's one
                            self.follow_path("after_sink")
                            self.state = "after_sink"
                        else: # go back outside if there's none
                            self.follow_path(("to_sink", True)) 
                            self.state = "go_outside"
                    else: # means the waiter is there to help
                        self.state = "working"

            elif self.state == "to_pass" or self.state == "going_back":
                self.step_along()
                if not self.path:  # arrived
                    if self.model.serving_queue:  # ✅ only pop if not empty
                        self.current_dish = self.model.serving_queue.popleft()

                        if self.current_dish.rice > self.model.rice:
                            self.model.serving_queue.append(self.current_dish)
                            if self.wait == 61: # waits for 1 minute
                                self.wait = 0
                                self.follow_path(("go_back", True))
                                self.state = "go_outside"
                            else:
                                self.reset_agent()
                                self.wait += 1
                                self.state = "to_pass"
                        else:
                            self.work = self.current_dish.serve
                            self.follow_path("go_serve")
                            self.state = "to_counter"
                    else:
                        if not (
                            any(chef.state != "idle" for chef in self.model.prep_chefs) and
                            any(cook.state != "idle" for cook in self.model.cook_chefs)
                        ) and (
                            len(self.model.prepTable) == 0 and len(self.model.cooking_queue) == 0
                        ):  
                            # If it's time to finish
                            self.follow_path("go_serve")
                            self.state = "go_finish"
                        else:
                            # Nothing to serve, waiter waits or idles
                            if self.wait == 61: # waits for 1 minute
                                self.wait = 0
                                self.follow_path(("go_back", True))
                                self.state = "go_outside"
                            else:
                                self.reset_agent()
                                self.wait += 1
                                self.state = "to_pass"

            elif self.state == "go_outside" or self.state == "go_finish":
                self.step_along()
                if not self.path:
                    if self.current_dish:
                        self.current_dish.serve_end = self.model.time
                        self.model.dishResults.append(self.current_dish)
                    self.reset_agent()

            elif self.state == "after_sink":
                self.step_along()
                if not self.path:  # arrived back from sink
                    self.reset_agent()
                    self.state = "to_pass" # start taking food from the pass table

            elif self.state == "to_counter":
                self.step_along()
                if not self.path:  # arrived
                    self.current_dish.serve_start = self.model.time
                    self.state = "working"

            elif self.state == "give_order":
                self.step_along()
                if not self.path:  # arrived
                    # recording dishes' arrival time when passed to the prep table
                    for order in self.orderList:
                        order.arrival_time = self.model.time
                    self.model.recievedOrder.extend(self.orderList) # pass it to the prep
                    self.reset_agent()
                    self.state = "idle"

            elif self.state == "outside_work":
                if self.work > 1:
                    self.work -= 1
                elif self.work == 1: # when agent is done working outside
                    self.reset_agent()

            elif self.state == "working":
                if self.work > 1:
                    self.work -= 1
                elif self.work == 1: # when agent is done working
                    # if agent was cleaning
                    if self.current_dish == "clean_up":
                        self.model.cleaningPile = 0
                        self.model.cleaning_in_progress -= 1   # free up slot

                        if self.model.serving_queue: # pick up dish if there's one
                            self.follow_path("after_sink")
                            self.state = "after_sink"
                        else: # go back outside if there's none
                            self.follow_path(("to_sink", True)) 
                            self.state = "go_outside"
                    # if agent was handling food
                    else:
                        # rice check
                        if self.current_dish.rice > self.model.rice:
                            self.current_dish.serve = 1
                            self.model.serving_queue.append(self.current_dish)
                            self.state = "idle"
                        else:
                            self.model.rice -= self.current_dish.rice
                            self.model.DirtyPlates += random.randint(3, 5)
                            self.follow_path("to_door")
                            self.state = "go_outside"
                    self.work = 0

        elif self.special == "dishwasher": #for dishwasheers
            if self.state == "idle":
                if (self.x, self.y) == self.spawn_cell: # if at spawn cell
                        self.follow_path(["to_pass", ("after_sink", True)])
                        self.state = "to_sink"
                elif self.model.cleaningPile >= 5:
                    self.work = 3
                    self.state = "working"

            elif self.state == "to_sink":
                self.step_along()
                if not self.path:  # arrived
                    if self.model.cleaningPile >= 5:
                        self.work = 3
                        self.state = "working"
                    else:
                        self.state = "idle"

            elif self.state == "working":
                if self.work > 1:
                    self.work -= 1
                elif self.work == 1: # when agent is done working
                    self.model.cleaningPile -= 5
                    self.work = 0
                    self.state = "idle"

##################################### Dish Blueprint #############################################
class Dish:
  def __init__(self, name, preptype, cooktype, prep_range, cook_range, rice, serve_range):
    self.name = name

    # dish metric records 
    self.arrival_time = None
    self.prep_start = None
    self.prep_end = None
    self.cook_start = None
    self.cook_end = None
    self.serve_start = None
    self.serve_end = None

    self.cook_queue_enter = None
    self.cook_wait_time = None

    self.serve_queue_enter = None
    self.serve_wait_time = None


    # setting the food's type
    self.preptype = preptype
    self.cooktype = cooktype

    # Workload ranges
    self.prep_range = prep_range  
    self.cook_range = cook_range
    self.serve_range = serve_range

    # the workload for each process
    self.prepare = 0 # for prep type chefs
    self.cook = 0 # for cook type chefs
    self.serve = 0 # for waiters

    # setting the amount of rice needed for this dish
    self.rice = rice
  
  # to properly show the name of the dish in printing
  def __str__(self):
    return self.name

# -----------------------------------------------------------------------------------------------------
#                                       Model Environment
# -----------------------------------------------------------------------------------------------------
class Kitchen(mesa.Model):
    def __init__(self, layout, menu, agentsVP, agentsMP, agentsPAR, agentsG, agentsF, agentsS, agentsCAR, agentsW, agentsDW, n_food):
        self.layout = layout
        self.menu = menu
        self.MAX_TIME = 4 * 60 * 60  # 4 hours in seconds
        self.time = 1
        self.n_food = n_food

        self.dishResults = []

        # calculate grid size from layout
        self.height = len(layout)       # number of rows
        self.width = len(layout[0])     # number of columns

        # Prep chef queues
        self.prepTable = deque()
        self.recievedOrder = deque()

        # Cook chef queues
        self.cooking_queue = deque()
        self.doneCooking = deque()

        # Waiter queues
        self.serving_queue = deque()
        self.fromOutside = deque()

        # basic simulated outside work - mimic waiter busy at outside
        self.outsideWork = {
            "serve_customer": 90,          # 1.5 minutes — walk to table, deliver food, small talk
            "take_order": 120,             # 2 minutes — chat, write order, confirm details
            "bring_drinks": 60,            # 1 minute — fetch and deliver drinks
            "clean_table": 180,            # 3 minutes — wipe, organize, reset
            "deliver_bill": 45,            # 45 seconds — hand over bill and wait briefly
            "collect_payment": 75,         # 1.25 minutes — handle cash/card, return change
            "chat_with_customers": 90,     # 1.5 minutes — occasional small talk or feedback
            "return_to_kitchen": 30        # 30 seconds — walking back inside
        }

        # kitchen resources
        self.rice = 50
        self.trash = 0
        self.DirtyPlates = 0

        self.cleaningPile = 0 # dishwasher's pile of plates to clean
        self.being_outside = [(4,11), (5,11), (7,11), (8,11)] # means the agent is outside

        # cleaning status
        self.cleaning_in_progress = 0   # number of waiters currently doing dishes
        self.max_cleaners = 1           # can tune to allow n waiters to go do the dishes

        self.takeback_plates = 0        # number of waiters currently bringing the dirty plates
        self.max_returnees = 1          # can tune to allow n waiters to bring the dishes

        self.throwing_trash = False     # only one allowed to throw the trash


        # scripting the arrival of orders
        self.orders = generate_simple_orders(menu=self.menu, total_orders=self.n_food, window_sec=self.MAX_TIME)
        for o in self.orders:
            o["arrival_time"] = round(o["arrival_time"])


        # agent spawn points with weights
        spawn_pools = {
            "prep_chef": [(7,4), (5,4), (5,3), (7,3), (5,5)],
            "cook_chef": [(2,6), (1,5), (1,3), (1,2), (1,4), (1,6)],
            "waiter":    [(3,10), (2,10), (1,9), (1,10)]
        }
        spawn_weights = {
            "prep_chef": [0.60, 0.20, 0.10, 0.05, 0.05],
            "cook_chef": [0.50, 0.20, 0.10, 0.10, 0.05, 0.05],
            "waiter":    [0.50, 0.30, 0.10, 0.10]
        }

        self.spawn_manager = SpawnManager(spawn_pools, spawn_weights) # spawning behavior

        # agent paths
        self.paths = {
            "prep_chef": {
                (7,4): {
                    "to_island": [(7,4), (7,3), (7,2), (6,2), (5,2), (4,2), (4,3)],
                    "go_back": [(4,3),(4,2),(5,2),(6,2),(7,2),(7,3),(7,4)],
                    "to_trash": [(4,3), (4,2), (5,2), (6,2), (6,1), (6,0)],
                    "after_trash": [(6,0), (6,1), (6,2), (7,2),(7,3),(7,4)],
                    "grab_ingred": [(7,4), (7,3), (7,2), (7,1), (7,0)]
                },
                (5,4): {
                    "to_island": [(5,4), (4,4)],
                    "go_back": [(4,4), (5,4)],
                    "to_trash": [(4,4), (5,4), (5,3), (5,2), (5,1), (5,0)],
                    "after_trash": [(5,0), (5,1), (5,2), (5,3), (5,4)],
                    "grab_ingred": [(5,4), (5,3), (5,2), (6,2), (7,2), (7,1)]
                },
                (5,3): {
                    "to_island": [(5,3), (4,3)],
                    "go_back": [(4,3), (5,3)],
                    "to_trash": [(4,3), (5,3), (5,2), (5,1), (5,0)],
                    "after_trash": [(5,0), (5,1), (5,2), (5,3)],
                    "grab_ingred": [(5,3), (5,2), (6,2), (7,2), (7,1), (7,0)]
                },
                (7,3): {
                    "to_island": [(7,3), (7,2), (6,2), (5,2), (4,2), (4,3)],
                    "go_back": [(4,3), (4,2), (5,2), (6,2), (7,2), (7,3)],
                    "to_trash": [(4,3), (4,2), (5,2), (6,2), (6,1), (6,0)],
                    "after_trash": [(6,0), (6,1), (6,2), (7,2), (7,3)],
                    "grab_ingred": [(7,3), (7,2), (7,1), (7,0)]
                },
                (5,5): {
                    "to_island": [(5,5), (4,5), (4,4)],
                    "go_back": [(4,4), (4,5), (5,5)],
                    "to_trash": [(4,4), (5,4), (5,3), (5,2), (5,1), (5,0)],
                    "after_trash": [(5,0), (5,1), (5,2), (5,3), (5,4), (5,5)],
                    "grab_ingred": [(5,5), (5,4), (5,3), (5,2), (6,2), (7,2), (7,1)]
                }
            },
            "cook_chef": {
                (2,6): {
                    "to_island": [(2,6), (2,5), (3,5)],
                    "go_cook": [(3,5), (2,5), (2,6)],
                    "to_pass": [(2,6), (3,6)],
                    "go_back": [(3,6), (3,5)],
                    "grab_rice": [(3,6), (4,6), (5,6), (6,6), (6,7), (6,8), (7,8)],
                    "cook_rice": [(7,8), (6,8), (6,7), (6,6), (5,6), (4,6), (3,6), (2,6)],
                    "to_prepA": [(2,6), (2,5), (3,5), (4,5), (4,4), (5,4)],
                    "to_prepB": [(3,6), (3,5), (4,5), (4,4), (5,4)],
                    "prep_to_island": [(5,4), (4,4), (4,5), (3,5)]
                },
                (1,5): {
                    "to_island": [(1,5), (2,5), (3,5)],
                    "go_cook": [(3,5), (2,5), (1,5)],
                    "to_pass": [(1,5), (2,5), (3,5), (3,6)],
                    "go_back": [(3,6), (3,5)],
                    "grab_rice": [(3,6), (4,6), (5,6), (6,6), (6,7), (6,8), (7,8)],
                    "cook_rice": [(7,8), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (3,5), (2,5), (1,5)],
                    "to_prepA": [(1,5), (2,5), (3,5), (4,5), (5,5)],
                    "to_prepB": [(3,6), (3,5), (4,5), (5,5)],
                    "prep_to_island": [(5,5), (4,5), (3,5)]
                },
                (1,3): {
                    "to_island": [(1,3), (2,3)],
                    "go_cook": [(2,3), (1,3)],
                    "to_pass": [(1,3), (2,3), (2,4), (2,5), (3,5), (4,5), (4,6)],
                    "go_back": [(4,6), (4,5), (3,5), (2,5), (2,4), (2,3)],
                    "grab_rice": [(4,6), (5,6), (6,6), (6,7), (6,8), (7,8)],
                    "cook_rice": [(7,8), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (3,5), (2,5), (2,4), (2,3), (1,3)],
                    "to_prepA": [(1,3), (2,3), (2,2), (3,2), (4,2), (5,2), (5,3)],
                    "to_prepB": [(4,6), (4,5), (4,4), (4,3), (5,3)],
                    "prep_to_island": [(5,3), (5,2), (4,2), (3,2), (2,2), (2,3)]
                },
                (1,2): {
                    "to_island": [(1,2), (2,2), (3,2)],
                    "go_cook": [(3,2), (2,2), (1,2)],
                    "to_pass": [(1,2), (2,2), (3,2), (4,2), (4,3), (4,4), (4,5), (4,6)],
                    "go_back": [(4,6), (4,5), (4,4), (4,3), (4,2), (3,2)],
                    "grab_rice": [(4,6), (5,6), (6,6), (6,7), (7,7)],
                    "cook_rice": [(7,7), (6,7), (6,6), (5,6), (4,6), (4,5), (4,4), (4,3), (4,2), (3,2), (2,2), (1,2)],
                    "to_prepA": [(1,2), (2,2), (3,2), (4,2), (4,3), (4,4), (5,4)],
                    "to_prepB": [(4,6), (4,5), (4,4), (5,4)],
                    "prep_to_island": [(5,4), (4,4), (4,3), (4,2), (3,2)]
                },
                (1,4): {
                    "to_island": [(1,4), (2,4)],
                    "go_cook": [(2,4), (1,4)],
                    "to_pass": [(1,4), (2,4), (2,5), (3,5), (4,5), (4,6)],
                    "go_back": [(4,6), (4,5), (3,5), (2,5), (2,4)],
                    "grab_rice": [(4,6), (5,6), (6,6), (6,7), (7,7)],
                    "cook_rice": [(7,7), (6,7), (6,6), (5,6), (4,6), (4,5), (3,5), (2,5), (2,4), (1,4)],
                    "to_prepA": [(1,4), (2,4), (2,5), (3,5), (4,5), (5,5)],
                    "to_prepB": [(4,6), (4,5), (5,5)],
                    "prep_to_island": [(5,5), (4,5), (3,5), (2,5), (2,4)]
                },
                (1,6): {
                    "to_island": [(1,6), (2,6), (2,5), (2,4)],
                    "go_cook": [(2,4), (2,5), (2,6), (1,6)],
                    "to_pass": [(1,6), (2,6), (2,5), (3,5), (4,5), (4,6)],
                    "go_back": [(4,6), (4,5), (3,5), (2,5), (2,4)],
                    "grab_rice": [(4,6), (5,6), (6,6), (6,7), (7,7)],
                    "cook_rice": [(7,7), (6,7), (6,6), (5,6), (4,6), (4,5), (3,5), (2,5), (2,6), (1,6)],
                    "to_prepA": [(1,6), (2,6), (2,5), (3,5), (4,5), (5,5)],
                    "to_prepB": [(4,6), (4,5), (5,5)],
                    "prep_to_island": [(5,5), (4,5), (3,5), (2,5), (2,4)]
                }
            },
            "waiter": { 
                (3,10): {
                    "to_pass": [(3,10), (4,10), (4,9)],
                    "go_serve": [(4,9), (4,10), (3,10)],
                    "to_door": [(3,10), (4,10), (5,10), (6,10), (7,10), (8,10), (8,11)],
                    "go_back": [(8,11), (8,10), (7,10), (6,10), (5,10), (4,10), (4,9)],
                    "to_sink": [(8,11), (8,10), (8,9), (7,9), (6,9), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (4,4), (4,3), (4,2), (4,1)],
                    "after_sink": [(4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (6,8), (6,9), (5,9), (4,9)],
                    "give_order": [(8,11), (8,10), (8,9), (7,9), (7,8), (7,7), (7,6)],
                    "after_order": [(7,6), (7,7), (7,8), (6,8), (6,9), (5,9), (4,9)]
                },
                (2,10): {
                    "to_pass": [(2,10), (2,9), (3,9), (4,9)],
                    "go_serve": [(4,9), (3,9), (2,9), (2,10)],
                    "to_door": [(2,10), (3,10), (4,10), (5,10), (6,10), (7,10), (7,11)],
                    "go_back": [(7,11), (7,10), (6,10), (5,10), (4,10), (4,9)],
                    "to_sink": [(7,11), (7,10), (7,9), (6,9), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (4,4), (4,3), (4,2), (3,2), (3,1)],
                    "after_sink": [(3,1), (3,2), (4,2), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (6,8), (6,9), (5,9), (4,9)],
                    "give_order": [(7,11), (7,10), (7,9), (7,8), (7,7), (7,6)],
                    "after_order": [(7,6), (7,7), (7,8), (6,8), (6,9), (5,9), (4,9)]
                },
                (1,9): {
                    "to_pass": [(1,9), (2,9), (3,9)],
                    "go_serve": [(3,9), (2,9), (1,9)],
                    "to_door": [(1,9), (2,9), (3,9), (4,9), (5,9), (5,10), (5,11)],
                    "go_back": [(5,11), (5,10), (5,9), (4,9), (3,9)],
                    "to_sink": [(5,11), (5,10), (5,9), (6,9), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (4,4), (4,3), (4,2), (3,2), (2,2), (2,1)],
                    "after_sink": [(2,1), (2,2), (3,2), (4,2), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (6,8), (6,9), (5,9), (4,9), (3,9)],
                    "give_order": [(5,11), (5,10), (5,9), (6,9), (6,8), (7,8), (7,7), (7,6)],
                    "after_order": [(7,6), (7,7), (7,8), (6,8), (6,9), (5,9), (4,9), (3,9)]
                },
                (1,10): {
                    "to_pass": [(1,10), (2,10), (2,9), (3,9)],
                    "go_serve": [(3,9), (2,9), (2,10), (1,10)],
                    "to_door": [(1,10), (2,10), (2,9), (3,9), (4,9), (4,10), (4,11)],
                    "go_back": [(4,11), (4,10), (4,9), (3,9)],
                    "to_sink": [(4,11), (4,10), (5,10), (5,9), (6,9), (6,8), (6,7), (6,6), (5,6), (4,6), (4,5), (4,4), (4,3), (4,2), (3,2), (2,2), (2,1), (1,1)],
                    "after_sink": [(1,1), (2,1), (2,2), (3,2), (4,2), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (6,8), (6,9), (5,9), (4,9), (3,9)],
                    "give_order": [(4,11), (4,10), (5,10), (5,9), (6,9), (6,8), (7,8), (7,7), (7,6)],
                    "after_order": [(7,6), (7,7), (7,8), (6,8), (6,9), (5,9), (4,9), (3,9)]
                }
            }
        }

        # agent lists
        self.prep_chefs = []
        self.cook_chefs = []
        self.waiters = []

        # IDs
        P_unique_id = 1
        C_unique_id = 1
        W_unique_id = 1


        # adding veggie preps
        if agentsVP > 0:
            for i in range(agentsVP):
                spawn_cell = self.spawn_manager.assign_cell("prep_chef")
                vP_chef = PrepChef(P_unique_id, self, 'Veggie_Prep')

                vP_chef.x, vP_chef.y = spawn_cell
                vP_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                vP_chef.paths = self.paths["prep_chef"][spawn_cell]
                self.prep_chefs.append(vP_chef)
                P_unique_id += 1

        # adding meat preps
        if agentsMP > 0:
            for i in range(agentsMP):
                spawn_cell = self.spawn_manager.assign_cell("prep_chef")
                mP_chef = PrepChef(P_unique_id, self, 'Meat_Prep')

                mP_chef.x, mP_chef.y = spawn_cell
                mP_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                mP_chef.paths = self.paths["prep_chef"][spawn_cell]
                self.prep_chefs.append(mP_chef)
                P_unique_id += 1

        # adding all-rounder preps
        if agentsPAR > 0:
            for i in range(agentsPAR):
                spawn_cell = self.spawn_manager.assign_cell("prep_chef")
                arP_chef = PrepChef(P_unique_id, self, 'all-rounder')

                arP_chef.x, arP_chef.y = spawn_cell
                arP_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                arP_chef.paths = self.paths["prep_chef"][spawn_cell]
                self.prep_chefs.append(arP_chef)
                P_unique_id += 1

        # adding Grillers
        if agentsG > 0:
            for i in range(agentsG):
                spawn_cell = self.spawn_manager.assign_cell("cook_chef")
                gC_chef = CookChef(C_unique_id, self, 'Grill')

                gC_chef.x, gC_chef.y = spawn_cell
                gC_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                gC_chef.paths = self.paths["cook_chef"][spawn_cell]
                self.cook_chefs.append(gC_chef)
                C_unique_id += 1

        # adding Fryers
        if agentsF > 0:
            for i in range(agentsF):
                spawn_cell = self.spawn_manager.assign_cell("cook_chef")
                fC_chef = CookChef(C_unique_id, self, 'Fry')

                fC_chef.x, fC_chef.y = spawn_cell
                fC_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                fC_chef.paths = self.paths["cook_chef"][spawn_cell]
                self.cook_chefs.append(fC_chef)
                C_unique_id += 1

        # adding Stewies
        if agentsS > 0:
            for i in range(agentsS):
                spawn_cell = self.spawn_manager.assign_cell("cook_chef")
                sC_chef = CookChef(C_unique_id, self, 'Stew')

                sC_chef.x, sC_chef.y = spawn_cell
                sC_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                sC_chef.paths = self.paths["cook_chef"][spawn_cell]
                self.cook_chefs.append(sC_chef)
                C_unique_id += 1

        # adding all-rounder cooks
        if agentsCAR > 0:
            for i in range(agentsCAR):
                spawn_cell = self.spawn_manager.assign_cell("cook_chef")
                arC_chef = CookChef(C_unique_id, self, 'all-rounder')

                arC_chef.x, arC_chef.y = spawn_cell
                arC_chef.spawn_cell = spawn_cell                     # <--- save spawn cell
                arC_chef.paths = self.paths["cook_chef"][spawn_cell]
                self.cook_chefs.append(arC_chef)
                C_unique_id += 1

        # adding servers
        if agentsW > 0:
            for i in range(agentsW):
                spawn_cell = self.spawn_manager.assign_cell("waiter")
                w = Waiter(W_unique_id, self, 'server')

                w.x, w.y = spawn_cell
                w.spawn_cell = spawn_cell                     # <--- save spawn cell
                w.paths = self.paths["waiter"][spawn_cell]
                self.waiters.append(w)
                W_unique_id += 1

        # adding dishwashers
        if agentsDW > 0:
            for i in range(agentsDW):
                spawn_cell = self.spawn_manager.assign_cell("waiter")
                dw = Waiter(W_unique_id, self, 'dishwasher')

                dw.x, dw.y = spawn_cell
                dw.spawn_cell = spawn_cell                     # <--- save spawn cell
                dw.paths = self.paths["waiter"][spawn_cell]
                self.waiters.append(dw)
                W_unique_id += 1

    def get_orders_at_tick(self, orders, current_tick):
        """Return all dishes for orders arriving at this tick."""
        arriving_dishes = []
        for order in orders:
            if order["arrival_time"] == current_tick:
                arriving_dishes.extend(order["dishes"])
        return arriving_dishes

    # ---------------- MOVEMENT ----------------
    def move(self):
        for chef in self.prep_chefs:
            chef.prepping_move()
        for cook in self.cook_chefs:
            cook.cooking_move()
        for waiter in self.waiters:
            waiter.serving_move()

        new_orders = self.get_orders_at_tick(self.orders, self.time)
        if len(new_orders) > 0:
            self.fromOutside.append(new_orders)

        for waiter in self.waiters:
            if random.random() < 0.15 and waiter.state == "idle" and waiter.special == "server":
                # randomly choose 1 to 3 tasks
                num_tasks = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

                # randomly pick that many tasks from the dictionary
                chosen_tasks = random.sample(list(self.outsideWork.items()), num_tasks)

                # sum up their total duration
                total_duration = sum(duration for _, duration in chosen_tasks)

                # assign total time as their "work" duration
                waiter.state = "outside_work"
                waiter.work = total_duration

        self.time += 1

    ############################# stopping rule #############################
    def check_list(self):
        """Return False when simulation should stop."""
        MAX_TIME = 4 * 60 * 60  # 4 hours in seconds

        # Stop if time limit reached
        if self.time >= MAX_TIME:
            return False

        # Queues empty?
        queues_empty = (
            len(self.orders) == 0 and
            len(self.prepTable) == 0 and
            len(self.cooking_queue) == 0 and
            len(self.serving_queue) == 0
        )

        # Any agents still working?
        agents_busy = (
            any(chef.state != "idle" for chef in self.prep_chefs) or
            any(cook.state != "idle" for cook in self.cook_chefs) or
            any(waiter.state != "idle" for waiter in self.waiters)
        )

        # Keep running if queues not empty or agents still busy
        return not (queues_empty and not agents_busy)

# -----------------------------------------------------------------------------------------------------
#                                       The Kitchen Layout
# -----------------------------------------------------------------------------------------------------
## ROOM FACTORS
# 0 = empty space (walkable)
# 1 = wall
# 2 = prep table
# 3 = stove
# 4 = serving counter
# 5 = cleaning area (sink and trash bin; they are close to each other in GBR)
# 6 = door (where waiters go out and serve the food)
# 7 = island counter (where prep chefs pass the prepared ingredients to the cook chefs)
# 8 = rice station (where rice stock is located and where cook chef get the rice)
# 9 = pass through table (where cook chefs pass the dish to the serving counter)
kitchen_layout = [
    [1,5,3,3,3,3,3,1,1,4,4,1],
    [5,0,0,0,0,0,0,3,4,0,0,4],
    [5,0,0,0,0,0,0,3,4,0,0,4],
    [5,0,0,7,7,0,0,3,4,0,0,4],
    [5,0,0,0,0,0,0,9,9,0,0,6],
    [6,0,0,0,0,0,0,9,9,0,0,6],
    [6,0,0,2,2,2,0,0,0,0,0,1],
    [0,0,0,0,0,2,0,0,0,0,0,6],
    [2,2,0,0,0,2,8,8,8,0,0,6]
]
# -----------------------------------------------------------------------------------------------------
#                                       Kitchen's Agent Spawning Behavior
# -----------------------------------------------------------------------------------------------------
class SpawnManager:
    def __init__(self, pools, weights):
        self.original_pools = pools
        self.original_weights = weights
        self.reset()

    def reset(self):
        self.available = {role: list(cells) for role, cells in self.original_pools.items()}

    def assign_cell(self, role):
        from random import choices

        # refill when empty
        if not self.available[role]:
            self.available[role] = list(self.original_pools[role])

        # filter weights only for available cells
        pool = self.available[role]
        weights = [self.original_weights[role][self.original_pools[role].index(cell)] for cell in pool]

        chosen = choices(pool, weights=weights, k=1)[0]
        self.available[role].remove(chosen)

        return chosen

# -----------------------------------------------------------------------------------------------------
#                                       Kitchen's Order Arrival Script
# -----------------------------------------------------------------------------------------------------
def generate_simple_orders(menu, total_orders=20, window_sec=4*60*60):
    orders = []

    # ---- WAVE CONFIG ----
    wave1_center = window_sec * 0.10   # early: 10% into time (feels like "start")
    wave2_center = window_sec * 0.85   # late: 85% into time (near the end)

    wave1_width = window_sec * 0.20    # wave1 lasts long (20% of total window)
    wave2_width = window_sec * 0.15    # wave2 lasts medium-long

    # how many orders each wave creates
    wave1_orders = int(total_orders * 0.65)  # big wave: 65% orders
    wave2_orders = total_orders - wave1_orders  # remaining go to wave2

    # ---- GENERATE WAVE 1 ----
    for oid in range(wave1_orders):
        t = random.gauss(mu=wave1_center, sigma=wave1_width)
        t = max(0, min(t, window_sec))

        num_dishes = random.randint(1, 3)
        dish_list = [clone_dish(random.choice(menu)) for _ in range(num_dishes)]

        orders.append({
            "order_id": oid,
            "arrival_time": t,
            "dishes": dish_list
        })

    # ---- GENERATE WAVE 2 ----
    for oid in range(wave1_orders, total_orders):
        t = random.gauss(mu=wave2_center, sigma=wave2_width)
        t = max(0, min(t, window_sec))

        num_dishes = random.randint(1, 3)
        dish_list = [clone_dish(random.choice(menu)) for _ in range(num_dishes)]

        orders.append({
            "order_id": oid,
            "arrival_time": t,
            "dishes": dish_list
        })

    # Sort final arrivals
    orders.sort(key=lambda x: x["arrival_time"])
    return orders

# create a new dish instance from a template dish
def clone_dish(template):
    dish = Dish(
        template.name,
        template.preptype,
        template.cooktype,
        template.prep_range,
        template.cook_range,
        template.rice,
        template.serve_range
    )

    # initialize the dynamic fields
    dish.arrival_time = None
    dish.prep_start = None
    dish.prep_end = None
    dish.cook_start = None
    dish.cook_end = None
    dish.serve_start = None
    dish.serve_end = None

    dish.cook_queue_enter = None
    dish.cook_wait_time = None
    dish.serve_queue_enter = None
    dish.serve_wait_time = None

    # select duration based on weights
    dish.prepare = random.choices(
        population=[d for d, w in dish.prep_range],
        weights=[w for d, w in dish.prep_range],
        k=1
    )[0]
    dish.cook = random.choices(
        population=[d for d, w in dish.cook_range],
        weights=[w for d, w in dish.cook_range],
        k=1
    )[0]
    dish.serve = random.choices(
        population=[d for d, w in dish.serve_range],
        weights=[w for d, w in dish.serve_range],
        k=1
    )[0]

    dish.prepare *= 60
    dish.cook *= 60
    dish.serve *= 60

    return dish

# -----------------------------------------------------------------------------------------------------
#                                       Dish Menu List
# -----------------------------------------------------------------------------------------------------

# Prep Types: Meat_Prep, Veggie_Prep
# Cook Types: Grill, Fry, Stew
# order of parameters: name | prep type | cook type | prepare load | cook load | rice needed | serve load
# 1 cup of rice = 5
# 1 minute → 60 seconds
# 2 minutes → 120 seconds
# 4 minutes → 240 seconds
# 5 minutes → 300 seconds
# 6 minutes → 360 seconds
# 7 minutes → 420 seconds
# 10 minutes → 600 seconds

# FRY
Buttered_Chicken = Dish("Buttered Chicken", "Meat_Prep", "Fry", [(1, 0.167), (2, 0.389), (3, 0.111), (4, 0.111), (5, 0.056), (6, 0.056), (7, 0.056)], 
                                                                [(8, 0.056), (9, 0.111), (10, 0.167), (11, 0.111), (12, 0.278), (13, 0.167), (15, 0.056)], 
                                                                5, [(1, 0.111), (2, 0.389), (3, 0.111), (4, 0.056), (5, 0.056), (6, 0.056), (11, 0.111), (15, 0.056)])
FishChicken_FingersNuggets = Dish("Fish/Chicken Fingers/Nuggets", "Meat_Prep", "Fry", [(3, 0.6), (8, 0.4)], [(4, 0.4), (5, 0.4), (6, 0.2)], 
                                                                                        5, [(5, 0.2), (10, 0.2), (13, 0.2), (14, 0.4)])
French_Fries = Dish("French Fries", "Veggie_Prep", "Fry", [(1, 0.188), (2, 0.375), (3, 0.375), (5, 0.0625)], 
                                                            [(5, 0.125), (6, 0.3125), (7, 0.0625), (8, 0.25), (9, 0.0625), 
                                                             (11, 0.0625), (12, 0.0625)], 
                                                             5, [(0, 0.0625), (1, 0.1875), (2, 0.125), (5, 0.1875), (6, 0.125), 
                                                                 (7, 0.125), (8, 0.0625), (17, 0.0625)])

# GRILL
Sisig = Dish("Sisig", "Meat_Prep", "Grill", [(0, 0.36), (3, 0.04), (4, 0.04), (7, 0.12), (8, 0.04), (10, 0.36)], 
                                            [(1,0.04),(2,0.16),(5,0.04),(6,0.12),(7,0.12),(8,0.12),(10,0.04),(12,0.04),(13,0.20),(15,0.04),(18,0.08)], 
                                            0, [(0,0.20),(1,0.40),(2,0.32)])
Ribs = Dish("Ribs", "Meat_Prep", "Grill", [(2, 0.364), (3, 0.182), (4, 0.182), (5, 0.091), (6, 0.091), (8, 0.091)], 
                                            [(4, 0.182), (5, 0.091), (6, 0.273), (10, 0.182), (12, 0.091), (13, 0.091), (15, 0.091)], 
                                            0, [(1, 0.091), (2, 0.364), (3, 0.091), (4, 0.091), (5, 0.091), (6, 0.091), (10, 0.091), (15, 0.091)])
Tapsilog = Dish("Tapsilog", "Meat_Prep", "Grill", [(1, 0.167), (2, 0.667), (3, 0.167)], [(5, 0.167), (6, 0.167), (7, 0.167), (8, 0.333), (9, 0.167)], 
                                                    0, [(0, 0.167), (2, 0.167), (4, 0.333), (7, 0.333)])
Chicken_Burger = Dish("Chicken Burger", "Meat_Prep", "Grill", [(1, 0.667), (2, 0.167), (5, 0.167)], 
                                                                [(3, 0.167), (4, 0.333), (7, 0.5)], 
                                                                0, [(1, 0.167), (2, 0.833)])

# STEW
Adobo = Dish("Adobo", "Meat_Prep", "Stew", [(1, 0.308), (2, 0.231), (3, 0.115), (4, 0.038), (5, 0.038), (7, 0.077), (13, 0.077), (14, 0.038)], 
                                            [(4, 0.037), (5, 0.037), (6, 0.037), (7, 0.111), (8, 0.074), (9, 0.148), (10, 0.185),
                                            (11, 0.037), (12, 0.111), (13, 0.037), (15, 0.037), (16, 0.074), (18, 0.074)], 
                                            0, [(1, 0.318), (2, 0.318), (3, 0.136), (6, 0.136), (8, 0.045), (23, 0.045)])
Guisado = Dish("Guisado", "Veggie_Prep", "Stew", [(1, 0.118), (2, 0.176), (3, 0.235), (5, 0.235), (6, 0.059), (9, 0.176)], 
                                                    [(5, 0.059), (8, 0.118), (9, 0.235), (10, 0.118), (11, 0.118), (13, 0.118), (15, 0.059), (18, 0.176)], 
                                                    0, [(1, 0.235), (2, 0.176), (3, 0.059), (4, 0.118), (5, 0.176), (8, 0.059), (17, 0.059), (24, 0.059)])
Ox_Tongue = Dish("Ox Tongue", "Meat_Prep", "Stew", [(0, 0.4), (1, 0.1), (4, 0.1), (6, 0.1), (10, 0.3)], 
                                                    [(1,0.2),(2,0.3),(6,0.1),(8,0.1),(11,0.1),(18,0.1),(24,0.1)], 
                                                    0, [(1,0.6),(2,0.4)])
Sizzling_Spaghetti = Dish("Sizzling Spaghetti", "Veggie_Prep", "Stew", [(0, 0.154), (1, 0.077), (2, 0.077), (4, 0.077), (10, 0.615)], 
                                                                        [(2, 0.538), (4, 0.077), (13, 0.154), (16, 0.077), (20, 0.077)], 
                                                                        0, [(0, 0.077), (1, 0.385), (2, 0.308), (3, 0.231)])

# adding all dishes to the menu
menu = [
    Buttered_Chicken,
    FishChicken_FingersNuggets,
    French_Fries,
    Sisig,
    Ribs,
    Tapsilog,
    Chicken_Burger,
    Adobo,
    Guisado,
    Ox_Tongue,
    Sizzling_Spaghetti
]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                                                            Streamlit UI / Main Method
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------
# Title and description
# -----------------------------
st.set_page_config(page_title="Kitchen Simulation", page_icon="🍳")

# CSS Hacking
# Custom CSS for gradient background
page_bg = """
<style>
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(135deg, #293b51, #429f97, #6a11cb, #2575fc);
    background-size: 400% 400%;
    animation: gradientAnimation 30s ease infinite;
}
div[data-testid="stForm"] {
    background-color: rgba(56, 62, 79, 0.95);
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("KMASS (Kitchen Modeling and Simulation System)")
st.write("📊 Enter agent numbers below, run the kitchen simulation, and see the results.")

#SPACER
st.write("")
st.write("")
#SPACER

# -----------------------------
# Streamlit Form: Simulation Setup
# -----------------------------
with st.form("simulation_form"):

    # --- Prep Chef Specialties ---
    st.subheader("Prep Chef Specialties 🔵")
    st.write("Agents who will prepare the ingredients before cooking the dishes")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        agents_VP = st.number_input("Veggie Prep🥦", min_value=0, max_value=5, value=1)
    with pcol2:
        agents_MP = st.number_input("Meat Prep🥩", min_value=0, max_value=5, value=1)
    with pcol3:
        agents_PAR = st.number_input("All-Rounder Prep👩‍🍳", min_value=1, max_value=5, value=1)

    # --- Cook Chef Specialties ---
    st.subheader("Cook Chef Specialties 🔴")
    st.write("Agents who will cook the dishes")
    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    with ccol1:
        agents_G = st.number_input("Grill🍖", min_value=0, max_value=5, value=1)
    with ccol2:
        agents_F = st.number_input("Fry🍟", min_value=0, max_value=5, value=1)
    with ccol3:
        agents_S = st.number_input("Stew🍲", min_value=0, max_value=5, value=1)
    with ccol4:
        agents_CAR = st.number_input("All-Rounder Cook👨‍🍳", min_value=1, max_value=5, value=1)

    # --- Waiter Specialties ---
    st.subheader("Waiter Specialties 🟡")
    st.write("Agents who will clean or assemble and serve the dishes")
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        agents_W = st.number_input("Server🤵‍♂️", min_value=1, max_value=10, value=3)
    with wcol2:
        agents_DW = st.number_input("Dish Washer🧼", min_value=0, max_value=5, value=1)

    # --- Simulation Conditions ---
    st.subheader("Simulation Conditions 🔧")
    st.write("💬 Higher Time Interval = Jumpier Animation / Lower Time Interval = Lower Simulation Coverage")
    st.write("💬 Each Order is a batch of dishes (number of dishes each batch is randomized)")
    scol1, scol2 = st.columns(2)
    with scol1:
        time_interval = st.slider(
            "Select time interval (seconds per frame) 🕑",
            min_value=1,
            max_value=60,
            value=5,
            step=1
        )
    with scol2:
        orders = st.number_input("Number of Orders 🍽", min_value=20, max_value=100, value=50)

    # --- Submit Button ---
    st.write("")  # spacer
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_clicked = st.form_submit_button("Run Simulation 🔥", use_container_width=True)

#SPACER
st.write("")
st.write("")
#SPACER

if run_clicked:
    def animation_generation():
        """Run 1 simulation - Only Create One Animation"""
        try:
            # -------------------------
            # Single Simulation Setup
            # -------------------------
            # initializing Kitchen
            kitchen = Kitchen(
                kitchen_layout, menu, 
                agents_VP, agents_MP, agents_PAR,
                agents_G, agents_F, agents_S,
                agents_CAR, agents_W, agents_DW,
                orders
            )
            # -----------------------------
            # Run simulation + collect history
            # -----------------------------
            history = []
            pending_history = [] # dishes that are recieved but not yet started preparing
            max_steps = 100000
            step = 0
            keep_going = True
            while keep_going and step < max_steps:
                step += 1
                kitchen.move()

                snapshot = []
                for chef in kitchen.prep_chefs:
                    snapshot.append((chef.x, chef.y, "Prep", chef.state))
                for cook in kitchen.cook_chefs:
                    snapshot.append((cook.x, cook.y, "Cook", cook.state))
                for waiter in kitchen.waiters:
                    snapshot.append((waiter.x, waiter.y, "Waiter", waiter.state))

                history.append(snapshot)
                pending_history.append(len(kitchen.recievedOrder))
                keep_going = kitchen.check_list()

            # -------------------------
            # Build animation
            # -------------------------
            layout_array = np.array(kitchen.layout)
            fig, ax = plt.subplots(figsize=(9, 9))

            # --- Add background image (behind grid & agents) ---
            bg_img = plt.imread("kitchen grid.png")  # path to the kitchen layout image
            bg_img = np.flipud(bg_img)   # flip vertically to display it right
            ax.imshow(
                bg_img,
                extent=[-0.5, kitchen.width - 0.5, -0.5, kitchen.height - 0.5],
                origin="upper",
                alpha=1.0  
            )
            # Assign integers in your layout_array for each furniture type
            # Example: 0 empty, 1 prep, 2 stove, 3 counter
            cmap = ListedColormap(["white", "black", "blue", "red", "cyan", "yellow", "pink", "orange", "purple", "yellowgreen"])

            # Draw layout (keep (0,0) top-left)
            ax.imshow(layout_array, cmap=cmap, origin="upper", alpha=0.25)
            ax.set_xlim(-0.5, kitchen.width - 0.5)
            ax.set_ylim(-0.5, kitchen.height - 0.5)
            ax.invert_yaxis() # correcting the orientation of the layout
            ax.set_title("Kitchen Simulation")

            # ---------------------------------
            # Agent color scheme (by role/state)
            # ---------------------------------
            role_colors = {
                "Prep": "deepskyblue",
                "Cook": "tomato",
                "Waiter": "gold"
            }
            state_effects = {
                "idle":   (0.4, 1.4),  # less saturated, slightly lighter
                "working": (2.2, 0.8), # highly saturated, slightly darker
            }
            # Initialize scatter points for each agent
            dots = []
            # Function to control agent's colors
            def adjust_color(color, sat_factor=1.0, light_factor=1.0):
                """Adjust color saturation and brightness/lightness."""
                r, g, b = mcolors.to_rgb(color)
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                s = max(0, min(1, s * sat_factor))   # increase saturation
                l = max(0, min(1, l * light_factor)) # adjust brightness
                r, g, b = colorsys.hls_to_rgb(h, l, s)
                return (r, g, b)

            # -------------------------
            # Initialize dots and text elements
            # -------------------------
            def init_animation(history, ax, role_colors, prep_cell=(6, 4)):
                # Create scatter dots for agents
                dots = []
                for x, y, role, state in history[0]:
                    base_color = role_colors.get(role, "gray")
                    dot = ax.scatter(y, x, s=400, color=base_color, edgecolor="black", linewidth=0.5)
                    dots.append(dot)

                # Fixed text for simulation time (corner)
                time_text = ax.text(
                    0.02, 0.95, "", transform=ax.transAxes,
                    fontsize=10, color="black", ha="left", va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3")
                )
                # Fixed text inside grid for prep count
                prep_text = ax.text(
                    prep_cell[1], prep_cell[0], "",  # (y, x) grid position
                    fontsize=10, color="black", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", boxstyle="round,pad=0.2")
                )
                return dots, time_text, prep_text

            # -------------------------
            # Update function
            # -------------------------
            def update_animation(frame, sampled_frames, dots, time_text, prep_text, role_colors, state_effects, pending_history, timeInterval):
                snapshot = sampled_frames[frame]

                # Update dots positions & colors
                for i, (x, y, role, state) in enumerate(snapshot):
                    base_color = role_colors.get(role, "gray")
                    sat, light = state_effects.get(state, (1.0, 1.0))
                    adjusted_color = adjust_color(base_color, sat_factor=sat, light_factor=light)

                    dots[i].set_offsets([y, x])
                    dots[i].set_color(adjusted_color)

                # Update simulation time text
                current_time = frame * timeInterval
                hours = int(current_time // 3600)
                minutes = int((current_time % 3600) // 60)
                seconds = int(current_time % 60)
                time_text.set_text(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

                # Update prep table text
                prep_text.set_text(f"{pending_history[frame]} pending")

                return dots + [time_text, prep_text]

            # -------------------------
            # Save to MP4 and display in Streamlit (with progress bar + 2 min cap)
            # -------------------------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:

                writer = FFMpegWriter(fps=5, metadata=dict(artist="KitchenSim"))

                max_duration_sec = 120  # 2 minutes
                max_frames = max_duration_sec * 5  # since fps=5
                # sample every nth frame
                sampled_frames = history[::time_interval]

                # Limit to 600 frames max
                sampled_frames = sampled_frames[:max_frames]
                total_frames = len(sampled_frames)

                writer.setup(fig, tmpfile.name, dpi=100)

                dots, time_text, prep_text = init_animation(history, ax, role_colors, prep_cell=(6, 4))

                for i in range(total_frames):
                    update_animation(
                        i, sampled_frames, dots, time_text, prep_text,
                        role_colors, state_effects, pending_history, time_interval
                    )
                    writer.grab_frame()

                writer.finish()
                video_path = tmpfile.name

            plt.close(fig)
            return {"status": "success", "video_path": video_path}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def metrics_generation():
        """Run 100 simulations - Collect and Calculate Metrics / Heatmaps"""

        try:
            ############################### HELPER FUNCTIONS FOR ERROR METRICS ##################################
            # --- Function to match dataset sizes using resampling --- #
            def match_sizes(real, sim):
                real = np.array(real)
                sim = np.array(sim)

                n_real = len(real)
                n_sim = len(sim)

                if n_real == n_sim:
                    return real, sim  # already same size

                # If simulation has fewer samples → upsample simulation
                if n_sim < n_real:
                    sim = np.random.choice(sim, size=n_real, replace=True)

                # If real data has fewer samples → upsample real data
                else:
                    real = np.random.choice(real, size=n_sim, replace=True)

                return real, sim
            # --- FUNCTION TO COMPUTE ERROR METRICS --- #
            def error_metrics(real, sim):
                real = np.array(real)
                sim = np.array(sim)

                mae = np.mean(np.abs(real - sim))
                rmse = np.sqrt(np.mean((real - sim) ** 2))
                return mae, rmse
            
            #################################### HEATMAP GENERATION SECTION #####################################
            num_runs = 100  # number of replications

            # Preparation for cumulative heatmaps
            layout_array = np.array(kitchen_layout)
            traffic_heatmap_total = np.zeros_like(layout_array, dtype=float)
            squeeze_heatmap_total = np.zeros_like(layout_array, dtype=float)
            ignore_Cells = [(1,1), (2,1), (3,1), (4,1), (4,11), (5,11), (7,11), (8,11), (7,6), (7,0), (7,1), (3,6), (4,6), (3,9), (4,9), (7,2)]

            ############################ TICK-BASED MODEL FOR HEATMAPS ############################
            for run in range(num_runs):
                # Initialize kitchen
                kitchen = Kitchen(
                    kitchen_layout, menu, 
                    agents_VP, agents_MP, agents_PAR,
                    agents_G, agents_F, agents_S,
                    agents_CAR, agents_W, agents_DW,
                    orders
                )
                history = []
                max_steps = 100000
                step = 0
                keep_going = True
                # run the simulation
                while keep_going and step < max_steps:
                    step += 1
                    kitchen.move()

                    snapshot = []
                    for chef in kitchen.prep_chefs:
                        snapshot.append((chef.x, chef.y, "Prep", chef.state))
                    for cook in kitchen.cook_chefs:
                        snapshot.append((cook.x, cook.y, "Cook", cook.state))
                    for waiter in kitchen.waiters:
                        snapshot.append((waiter.x, waiter.y, "Waiter", waiter.state))
                    history.append(snapshot)
                    keep_going = kitchen.check_list()

                # ---------- Heatmaps ----------
                traffic_heatmap = np.zeros_like(layout_array, dtype=float)
                squeeze_heatmap = np.zeros_like(layout_array, dtype=float)

                # Traffic
                for snapshot in history:
                    prep_idx, cook_idx, waiter_idx = 0, 0, 0
                    for (x, y, role, state) in snapshot:
                        if role == "Prep":
                            agent = kitchen.prep_chefs[prep_idx]; prep_idx += 1
                        elif role == "Cook":
                            agent = kitchen.cook_chefs[cook_idx]; cook_idx += 1
                        elif role == "Waiter":
                            agent = kitchen.waiters[waiter_idx]; waiter_idx += 1
                        if (x, y) != agent.spawn_cell and (x, y) not in ignore_Cells:
                            traffic_heatmap[x, y] += 1
                # Squeeze
                for snapshot in history:
                    cell_counter = {}
                    prep_idx, cook_idx, waiter_idx = 0, 0, 0
                    for (x, y, role, state) in snapshot:
                        if role == "Prep":
                            agent = kitchen.prep_chefs[prep_idx]; prep_idx += 1
                        elif role == "Cook":
                            agent = kitchen.cook_chefs[cook_idx]; cook_idx += 1
                        elif role == "Waiter":
                            agent = kitchen.waiters[waiter_idx]; waiter_idx += 1
                        if (x, y) == agent.spawn_cell or (x, y) in ignore_Cells:
                            continue
                        cell_counter[(x, y)] = cell_counter.get((x, y), 0) + 1
                    for (x, y), count in cell_counter.items():
                        if count > 1:
                            squeeze_heatmap[x, y] += count

                # Accumulate for averaging
                traffic_heatmap_total += traffic_heatmap
                squeeze_heatmap_total += squeeze_heatmap
                # Free up memory
                del history
            # ---------- Average + Normalize Heatmaps ----------
            traffic_heatmap_total /= num_runs
            squeeze_heatmap_total /= num_runs

            if traffic_heatmap_total.max() > 0:
                traffic_heatmap_total /= traffic_heatmap_total.max()
            if squeeze_heatmap_total.max() > 0:
                squeeze_heatmap_total /= squeeze_heatmap_total.max()

            # ---------- Plot Heatmaps ----------
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ax1.imshow(layout_array, cmap="gray", origin="upper")
            furniture_mask = (layout_array > 0) & (layout_array != 6)
            ax1.imshow(furniture_mask, cmap="Greys", origin="upper")
            cmap1 = LinearSegmentedColormap.from_list("traffic_cmap", ["blue", "red"])
            hm1 = ax1.imshow(traffic_heatmap_total, cmap=cmap1, alpha=0.6, origin="upper")
            plt.colorbar(hm1, label="Avg Traffic Intensity", ax=ax1)
            ax1.set_title("Average Traffic Heatmap over 100 Runs")

            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.imshow(layout_array, cmap="gray", origin="upper")
            ax2.imshow(furniture_mask, cmap="Greys", origin="upper")
            cmap2 = LinearSegmentedColormap.from_list("squeeze_cmap", ["green", "red"])
            hm2 = ax2.imshow(squeeze_heatmap_total, cmap=cmap2, alpha=0.6, origin="upper")
            plt.colorbar(hm2, label="Avg Squeeze Intensity", ax=ax2)
            ax2.set_title("Average Squeeze Heatmap over 100 Runs")

    
            ############################## PERFORMANCE METRICS GENERATION SECTION ##################################
            # --- REAL KITCHEN DATA --- #
            WT_minutes = [14, 2, 5, 8, 1, 5, 0, 11, 2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 1, 2, 1, 3, 0, 1, 0, 0, 4, 
                    5, 0, 0, 1, 0, 2, 0, 7, 11, 0, 1, 7, 1, 0, 0, 13, 2, 0, 3, 5, 0, 6, 0, 5, 5, 0, 0, 3, 1, 
                    2, 0, 5, 2, 0, 1, 2, 2, 0, 1, 3, 3, 2, 4, 5, 2, 5, 2, 4, 0, 4, 2, 1, 2, 1, 0, 1, 1, 5, 2, 2, 1, 2, 
                    13, 3, 11, 2, 10, 1, 3, 2, 1, 1, 0, 9, 5, 3, 5, 2, 0, 1, 3, 5, 2, 2, 3, 4, 5, 
                    10, 8, 10, 1, 4, 2, 5, 7, 4, 2, 0, 2, 2, 9, 13, 16, 5, 14, 7, 0, 0, 1, 1]  # list of actual waiting times
            ST_minutes = [8, 15, 12, 12, 16, 16, 15, 27, 21, 21, 21, 15, 7, 14, 6, 10, 13, 21, 10, 15, 15, 16, 14, 6, 14, 6, 
                    12, 13, 13, 19, 9, 13, 8, 17, 8, 38, 34, 19, 37, 9, 14, 25, 20, 24, 13, 16, 18, 16, 25, 16, 2, 14,
                    16, 8, 17, 20, 15, 20, 20, 14, 14, 7, 7, 14, 17, 6, 14, 20, 13, 14, 15, 11, 14, 22, 14, 14, 13, 16, 13, 
                    33, 32, 18, 19, 17, 17, 18, 9, 13, 21, 27, 16, 25, 18, 23, 21, 5, 16, 14, 12, 10, 23, 14, 18, 
                    22, 17, 19, 8, 13, 10, 15, 12, 14, 17, 20, 14, 21, 21, 16, 13, 13, 15, 19, 19, 12, 8, 9, 10, 
                    14, 26, 25, 19, 29, 24, 13, 9, 14, 13, 13]  # list of actual service times
            TAT_minutes = [22, 17, 17, 20, 17, 21, 15, 38, 23, 23, 23, 17, 9, 16, 7, 12, 13, 23, 12, 16, 17, 17, 17, 6, 
                        15, 6, 12, 17, 18, 19, 9, 14, 8, 19, 8, 45, 45, 19, 38, 16, 15, 25, 20, 37, 15, 16, 21, 21, 
                        25, 22, 2, 19, 21, 8, 17, 23, 16, 22, 20, 19, 16, 8, 8, 16, 19, 6, 15, 23, 16, 16, 19, 16, 16, 27, 16, 
                        18, 13, 20, 15, 34, 34, 19, 19, 18, 18, 23, 11, 15, 22, 29, 29, 28, 29, 25, 31, 6, 19, 16, 13, 11, 23, 
                        23, 23, 25, 22, 21, 8, 14, 13, 20, 14, 16, 20, 24, 19, 31, 29, 26, 14, 17, 17, 24, 26, 16, 10, 9, 12, 
                        16, 35, 38, 35, 34, 38, 20, 9, 14, 14, 14] # list of actual turnaround times

            # --- SIMULATION DATA --- #
            sim_wt = []   # waiting times from simulation
            sim_st = []   # service times (prep + cook + serve) from simulation
            sim_tat = []  # turnaround times from simulation

            throughputs = [] # "number of dish served per hour"

            staff_utils = []  # store staff utilization per run
            busy_states = ["busy", "outside_work", "cleaning", "collecting_plates", "cooking_rice"]

            ############################ EVENT-DRIVEN MODEL FOR PERFORMANCE METRICS ############################
            # good seeds: 39, 
            random.seed(39)
            np.random.seed(39)
            for i in range(num_runs):
            # Create your kitchen simulation
                kitchen2 = ED_Kitchen(
                    menu, agents_VP, agents_MP, agents_PAR,
                    agents_G, agents_F, agents_S,
                    agents_CAR, agents_W, agents_DW, orders
                    )
                kitchen2.run()

                # Calculate staff utilization directly
                staff_utilization = {}

                # --- Prep Utilization (normal agents) ---
                for agent in kitchen2.prep_chefs:
                    if agent.state in busy_states:
                        agent.total_busy_time += kitchen2.MAX_TIME - agent.last_state_change

                    label = f"{agent.name} ({agent.specialization})"
                    staff_utilization[label] = agent.total_busy_time / kitchen2.MAX_TIME

                # --- Cook Utilization (group every 3 cooks = 1 real cook) ---
                cook_agents = kitchen2.cook_chefs
                group_size = 3
                num_real_cooks = len(cook_agents) // group_size

                for i in range(num_real_cooks):
                    group = cook_agents[i*group_size : (i+1)*group_size]

                    total_busy = 0
                    for agent in group:
                        if agent.state in busy_states:
                            agent.total_busy_time += kitchen2.MAX_TIME - agent.last_state_change
                        total_busy += agent.total_busy_time

                    # A real cook has capacity of 3 cooks → denominator = 3 * MAX_TIME
                    real_cook_util = total_busy / (3 * kitchen2.MAX_TIME)

                    spec = group[0].specialization        # take the specialization of the first cook
                    label = f"Cook{i+1} ({spec})"

                    staff_utilization[label] = real_cook_util

                # --- Waiter Utilization (normal agents) ---
                for agent in kitchen2.waiters:
                    if agent.state in busy_states:
                        agent.total_busy_time += kitchen2.MAX_TIME - agent.last_state_change

                    label = f"{agent.name} ({agent.special})"
                    staff_utilization[label] = agent.total_busy_time / kitchen2.MAX_TIME

                # Store results
                staff_utils.append(staff_utilization)

                for dish in kitchen2.finished_dishes:
                    if None in [dish.arrival_time, dish.prep_start, dish.prep_end,
                                dish.cook_start, dish.cook_end, dish.serve_start, dish.serve_end]:
                        continue
                    #--- FIX TIMESTAMP ORDER IF WRONG ---
                    if dish.prep_end < dish.prep_start:
                        dish.prep_start, dish.prep_end = dish.prep_end, dish.prep_start

                    if dish.cook_end < dish.cook_start:
                        dish.cook_start, dish.cook_end = dish.cook_end, dish.cook_start

                    if dish.serve_end < dish.serve_start:
                        dish.serve_start, dish.serve_end = dish.serve_end, dish.serve_start

                    prep_duration = dish.prep_end - dish.prep_start
                    cook_duration = dish.cook_end - dish.cook_start
                    serve_duration = dish.serve_end - dish.serve_start

                    waiting_time = dish.prep_start - dish.arrival_time
                    # new waiting times (queue delays)
                    cook_wait = dish.cook_wait_time if dish.cook_wait_time is not None else 0
                    serve_wait = dish.serve_wait_time if dish.serve_wait_time is not None else 0

                    waiting_time += cook_wait
                    waiting_time += serve_wait

                    tat = dish.serve_end - dish.arrival_time

                    # sum of all stages + waiting
                    st = prep_duration + cook_duration + serve_duration

                    # convert into minutes of time unit
                    waiting_time /= 60
                    st /= 60
                    tat /= 60

                    sim_wt.append(waiting_time)
                    sim_st.append(st)
                    sim_tat.append(tat)
            
                # Throughput calculation
                sim_hours = 4
                finished_dishes = len(kitchen2.finished_dishes)

                dishes_per_hour = finished_dishes / sim_hours
                throughputs.append(dishes_per_hour)

            # ---------------------------------- Aggregate Metrics -------------------------------------
            # Collect the data into a dictionary
            all_metrics = {
                "Avg_Wait": sim_wt,
                "Avg_Service": sim_st,
                "Avg_TaT": sim_tat,
            }
            # Check if there is any data
            if not any(all_metrics.values()):
                return {"status": "no_data", "summary": None}

            df = pd.DataFrame(all_metrics)
            summary = {}

            for col in ["Avg_Service", "Avg_TaT", "Avg_Wait"]:
                mean = df[col].mean()
                std = df[col].std()
                ci_low, ci_high = stats.t.interval(0.95, len(df[col]) - 1, loc=mean, scale=stats.sem(df[col]))
                summary[col] = {
                    "Mean": round(mean, 2),
                    "StdDev": round(std, 2),
                    "95% CI Low": round(ci_low, 2),
                    "95% CI High": round(ci_high, 2)
                }
            # Add throughput to the summary table
            throughput_mean = np.mean(throughputs)
            throughput_std = np.std(throughputs, ddof=1)
            throughput_ci = stats.t.interval(0.95, len(throughputs)-1, loc=throughput_mean, scale=stats.sem(throughputs))
            summary["Throughput"] = {
                "Mean": round(throughput_mean,2),
                "StdDev": round(throughput_std,2),
                "95% CI Low": round(throughput_ci[0],2),
                "95% CI High": round(throughput_ci[1],2)
            }
            summary_df = pd.DataFrame(summary).T

            # -------------------- BAR CHART WITH CI --------------------
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            # Metrics
            metrics = ["WT", "ST", "TAT"]

            # Means
            means_real = [np.mean(WT_minutes), np.mean(ST_minutes), np.mean(TAT_minutes)]
            means_sim = [np.mean(sim_wt), np.mean(sim_st), np.mean(sim_tat)]

            # Standard errors
            se_real = [stats.sem(WT_minutes), stats.sem(ST_minutes), stats.sem(TAT_minutes)]
            se_sim = [stats.sem(sim_wt), stats.sem(sim_st), stats.sem(sim_tat)]

            x = np.arange(len(metrics))
            width = 0.35

            # Bar plots
            rects1 = ax_bar.bar(x - width/2, means_real, width, yerr=se_real, capsize=6, label="Real", color="hotpink")
            rects2 = ax_bar.bar(x + width/2, means_sim, width, yerr=se_sim, capsize=6, label="Sim", color="mediumturquoise")

            ax_bar.set_ylabel("Time (minutes)")
            ax_bar.set_title("Comparison of Avg WT, ST, TAT (Real vs Sim)")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(metrics)
            ax_bar.legend()
            ax_bar.grid(axis="y", linestyle="--", alpha=0.7)

            # -------------------- VERTICAL BAR CHART STAFF UTILIZATION --------------------
            # Convert list of dicts → DataFrame
            df_utils = pd.DataFrame(staff_utils)  # rows=runs, columns=agents

            # Average utilization per agent across runs
            avg_utils = df_utils.mean(axis=0)

            # Assign colors based on agent type
            colors = []
            for name in avg_utils.index:
                if "PrepChef" in name:
                    colors.append("turquoise")
                elif "Waiter" in name:
                    colors.append("gold")
                elif "Cook" in name:
                    colors.append("orangered")
                else:
                    colors.append("gray")  # fallback

            # --- Plot Bar Chart ---
            staff_bar, ax = plt.subplots(figsize=(10,5))
            ax.barh(avg_utils.index, avg_utils.values, color=colors)
            ax.set_ylabel("Average Utilization")
            ax.set_title("Average Staff Utilization Across Runs")
            plt.tight_layout()

            # agent specialties
            cookChef_specialties = [agent.special for agent in kitchen.cook_chefs]
            prepChef_specialties = [agent.special for agent in kitchen.prep_chefs]
            waiter_specialties = [agent.special for agent in kitchen.waiters]

            ############################ ERROR METRICS GENERATION ############################
            # --- SIMULATION DATA FOR ERROR --- #
            esim_wt = []
            esim_st = []
            esim_tat = []
            # run ED simulation for results validation
            for i in range(4):
                ekitchen = ED_Kitchen(menu=menu)
                ekitchen.run()  # run your event-driven simulation

                for dish in ekitchen.finished_dishes:
                    if None in [dish.arrival_time, dish.prep_start, dish.prep_end,
                                dish.cook_start, dish.cook_end, dish.serve_start, dish.serve_end]:
                        continue
                    #--- FIX TIMESTAMP ORDER IF WRONG ---
                    if dish.prep_end < dish.prep_start:
                        dish.prep_start, dish.prep_end = dish.prep_end, dish.prep_start

                    if dish.cook_end < dish.cook_start:
                        dish.cook_start, dish.cook_end = dish.cook_end, dish.cook_start

                    if dish.serve_end < dish.serve_start:
                        dish.serve_start, dish.serve_end = dish.serve_end, dish.serve_start

                    prep_duration = dish.prep_end - dish.prep_start
                    cook_duration = dish.cook_end - dish.cook_start
                    serve_duration = dish.serve_end - dish.serve_start

                    waiting_time = dish.prep_start - dish.arrival_time
                    # new waiting times (queue delays)
                    cook_wait = dish.cook_wait_time if dish.cook_wait_time is not None else 0
                    serve_wait = dish.serve_wait_time if dish.serve_wait_time is not None else 0

                    waiting_time += cook_wait
                    waiting_time += serve_wait

                    tat = dish.serve_end - dish.arrival_time

                    # sum of all stages + waiting
                    st = prep_duration + cook_duration + serve_duration

                    esim_wt.append(waiting_time)
                    esim_st.append(st)
                    esim_tat.append(tat)

            # Convert minutes into seconds
            real_wt = [x * 60 for x in WT_minutes]
            real_st = [x * 60 for x in ST_minutes]
            real_tat = [x * 60 for x in TAT_minutes]

            # Convert to numpy
            ereal_wt = np.array(real_wt)
            ereal_st = np.array(real_st)
            ereal_tat = np.array(real_tat)
            esim_wt = np.array(esim_wt)
            esim_st = np.array(esim_st)
            esim_tat = np.array(esim_tat)

            # --- MATCH SIZES BEFORE COMPUTING METRICS --- #
            real_wt_m, sim_wt_m = match_sizes(ereal_wt, esim_wt)
            real_st_m, sim_st_m = match_sizes(ereal_st, esim_st)
            real_tat_m, sim_tat_m = match_sizes(ereal_tat, esim_tat)

            # --- COMPUTE ERRORS --- #
            wt_mae, wt_rmse = error_metrics(real_wt_m, sim_wt_m)
            st_mae, st_rmse = error_metrics(real_st_m, sim_st_m)
            tat_mae, tat_rmse = error_metrics(real_tat_m, sim_tat_m)

            # --- STATISTICAL TESTS (now same size) --- #
            wt_t, wt_p = ttest_ind(sim_wt_m, real_wt_m)
            st_t, st_p = ttest_ind(sim_st_m, real_st_m)
            tat_t, tat_p = ttest_ind(sim_tat_m, real_tat_m)

            # --- 1. MAE & RMSE Bar Chart ---
            error_df = pd.DataFrame({
                "Metric": ["WT", "ST", "TaT"],
                "MAE": [wt_mae, st_mae, tat_mae],
                "RMSE": [wt_rmse, st_rmse, tat_rmse]
            })
            fig_error, ax_error = plt.subplots(figsize=(6,4))
            error_df.set_index("Metric")[["MAE","RMSE"]].plot(kind="bar", ax=ax_error, rot=0)
            ax_error.set_ylabel("Error (seconds)")
            ax_error.set_title("MAE and RMSE per Metric")
            plt.tight_layout()

            # --- 2. t-test p-values Heatmap ---
            pvals_df = pd.DataFrame({
                "Metric": ["WT", "ST", "TaT"],
                "p-value": [wt_p, st_p, tat_p]
            }).set_index("Metric")
            fig_ttest, ax_ttest = plt.subplots(figsize=(4,3))
            sns.heatmap(pvals_df, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, ax=ax_ttest)
            ax_ttest.set_title("t-test p-values")
            plt.tight_layout()

            return {
                "status": "success",
                "summary": summary_df,
                "traffic_heatmap": fig1,
                "squeeze_heatmap": fig2,
                "cook_special": cookChef_specialties,
                "prep_special": prepChef_specialties,
                "waiter_special": waiter_specialties,
                "metrics_barchart": fig_bar,
                "staff_barchart": staff_bar,
                "error_barchart": fig_error,
                "ttest_heatmap": fig_ttest,
                "error_table": error_df
            }
        
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def round_nearest(x):
        return int(x + 0.5)
    

    # ------------------------------------------------------------------------------------------
    #                                    MAIN (Threads Master)
    # ------------------------------------------------------------------------------------------
    # Two progress bars and text placeholders
    anim_text = st.empty()
    anim_bar = st.progress(0)
    met_text = st.empty()
    met_bar = st.progress(0)

    animation_result = {}
    metrics_result = {}
    animation_done = threading.Event()
    metrics_done = threading.Event()

    def run_animation_thread():
        try:
            animation_result["data"] = animation_generation()
            animation_done.set()
        except Exception as e:
            animation_result["data"] = {"status": "failed", "error": str(e)}
            animation_done.set()

    def run_metrics_thread():
        try:
            metrics_result["data"] = metrics_generation()
            metrics_done.set()
        except Exception as e:
            metrics_result["data"] = {"status": "error", "error": str(e)}
            metrics_done.set()

    # Start both threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_animation_thread)
        executor.submit(run_metrics_thread)

        anim_progress = 0
        met_progress = 0

        while not (animation_done.is_set() and metrics_done.is_set()):
            # ---------------- Animation progress simulation ----------------
            if not animation_done.is_set():
                if anim_progress < 60:
                    anim_progress += 0.6
                elif anim_progress < 90:
                    anim_progress += 0.3
                else:
                    anim_progress += 0.1
                anim_progress = min(anim_progress, 99)
                anim_bar.progress(int(anim_progress))
                anim_text.text(f"🎬 Generating animation... ({int(anim_progress)}%)")

                if anim_progress >= 98:
                    anim_text.text("🎞️ Finalizing video... please wait a moment")
            else:
                anim_bar.progress(100)
                anim_text.text("✅ Animation complete!")

            # ---------------- Metrics progress simulation ----------------
            if not metrics_done.is_set():
                if met_progress < 70:
                    met_progress += 1.2
                elif met_progress < 90:
                    met_progress += 0.6
                else:
                    met_progress += 0.3
                met_progress = min(met_progress, 99)
                met_bar.progress(int(met_progress))
                met_text.text(f"📊 Calculating metrics... ({int(met_progress)}%)")
            else:
                met_bar.progress(100)
                met_text.text("✅ Metrics complete!")

            time.sleep(0.3)

    # Final 100% once both threads finish
    anim_bar.progress(100)
    met_bar.progress(100)
    anim_text.text("✅ Animation complete!")
    met_text.text("✅ Metrics complete!")

    # --------------------------------------------------------------------------------------
    #                                    STREAMLIT DISPLAY RESULTS
    # --------------------------------------------------------------------------------------
    if "data" in animation_result and "data" in metrics_result:

        a_data = animation_result["data"]
        m_data = metrics_result["data"]

        if a_data["status"] == "success" and m_data["status"] == "success":
            st.success("✅ Simulation Complete! Displaying all results")

            # --- Animation Output ---
            st.subheader("Simulation Animation 🎬")
            st.write("💬 Pale color means idle, Normal color means moving, and Oversaturated means working.")
            st.video(a_data["video_path"])  # Displays the generated MP4 video

            # --- Worker Salary ---
            worker_num = len(m_data["cook_special"])
            worker_num += len(m_data["prep_special"])
            worker_num += len(m_data["waiter_special"])
            money = (worker_num * 450) * 7
            st.subheader(f"💵Total Worker Salary per Week:  ₱{money:,}")

            # throughput interpretation
            num_TP = round_nearest(m_data["summary"].at["Throughput", "Mean"])
            st.subheader(f"☑️Kitchen serves:    {num_TP} dishes per hour")

            # --- Agent Specialties ---
            st.subheader("Agent Specialties")
            st.write("🔪 Prep Chefs:", m_data["prep_special"])
            st.write("👨‍🍳 Cook Chefs:", m_data["cook_special"])
            st.write("🤵‍♀️ Waiters:", m_data["waiter_special"])

            # --- Metrics ---
            st.subheader("Performance Metrics (100 Runs)")
            st.dataframe(m_data["summary"])
            st.subheader("💬 Metrics Interpretation")
            # simple interpretation code:
            high_ST = round_nearest(m_data["summary"].at["Avg_Service", "95% CI High"])
            low_ST = round_nearest(m_data["summary"].at["Avg_Service", "95% CI Low"])
            high_TAT = round_nearest(m_data["summary"].at["Avg_TaT", "95% CI High"])
            low_TAT = round_nearest(m_data["summary"].at["Avg_TaT", "95% CI Low"])
            high_WT = round_nearest(m_data["summary"].at["Avg_Wait", "95% CI High"])
            low_WT = round_nearest(m_data["summary"].at["Avg_Wait", "95% CI Low"])

            rareST = round_nearest(m_data["summary"].at["Avg_Service", "StdDev"])
            rareTAT = round_nearest(m_data["summary"].at["Avg_TaT", "StdDev"])
            rareWT = round_nearest(m_data["summary"].at["Avg_Wait", "StdDev"])

            interpretST = "- Service Time is mostly around"
            interpretTAT = "- Turnaround Time is mostly around"
            interpretWT = "- Waiting Time is mostly around"

            # Service Time
            if high_ST == low_ST:
                interpretST = f"{interpretST} {high_ST} minutes"
            else:
                interpretST = f"{interpretST} {low_ST}-{high_ST} minutes"
            # Turnaround Time 
            if high_TAT == low_TAT:
                interpretTAT = f"{interpretTAT} {high_TAT} minutes"
            else:
                interpretTAT = f"{interpretTAT} {low_TAT}-{high_TAT} minutes"
            # Waiting Time
            if high_WT == low_WT:
                interpretWT = f"{interpretWT} {high_WT} minutes"
            else:
                interpretWT = f"{interpretWT} {low_WT}-{high_WT} minutes"


            if rareST > high_ST:
                if (rareST - high_ST) >= 8:
                    interpretST = f"{interpretST} with a rare slow scenario of {rareST} minutes"
            elif rareST < low_ST:
                if (low_ST - rareST) >= 8:
                    interpretST = f"{interpretST} with a rare fast scenario of {rareST} minutes"

            if rareTAT > high_TAT:
                if (rareTAT - high_TAT) >= 8:
                    interpretTAT = f"{interpretTAT} with a rare slow scenario of {rareTAT} minutes"
            elif rareTAT < low_TAT:
                if (low_TAT - rareTAT) >= 8:
                    interpretTAT = f"{interpretTAT} with a rare fast scenario of {rareTAT} minutes"

            if rareWT > high_WT:
                if (rareWT - high_WT) >= 8:
                    interpretWT = f"{interpretWT} with a rare slow scenario of {rareWT} minutes"
            elif rareWT < low_WT:
                if (low_WT - rareWT) >= 8:
                    interpretWT = f"{interpretWT} with a rare fast scenario of {rareWT} minutes"

            st.write(interpretST)
            st.write(interpretTAT)
            st.write(interpretWT)

            #SPACER
            st.write("")
            st.write("")
            #SPACER

            # --- Bar chart for main metrics ---
            st.pyplot(m_data["metrics_barchart"])
            st.write("💬 Higher bar means longer amounts of minutes (slower).")
            st.write("💬 Lower bar means shorter amounts of minutes (faster).")
            st.write("")
            st.subheader("Staff Utilization")
            st.pyplot(m_data["staff_barchart"])

            #SPACER
            st.write("")
            st.write("")
            st.write("")
            #SPACER

            # --- Heatmaps ---
            st.subheader("Average Traffic Heatmap 🚶‍♂️‍➡️")
            st.write("💬 Where staff usually walks through most of the time.")
            st.pyplot(m_data["traffic_heatmap"])

            st.subheader("Average Squeeze Heatmap 👥")
            st.write("💬 Where staff usually squeeze through each other.")
            st.pyplot(m_data["squeeze_heatmap"])

            #SPACER
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            #SPACER

            # --- ERROR METRICS ---
            st.subheader("Error Metrics 🎯")
            st.write("💬 Results validation of how accurate the simulation is.")
            st.pyplot(m_data["ttest_heatmap"])
            st.write("")
            st.write("💬 MAE and RMSE (additional information)")
            m_data["error_table"]["MAE"] /= 60
            m_data["error_table"]["RMSE"] /= 60
            st.dataframe(m_data["error_table"])
            st.pyplot(m_data["error_barchart"])

        else:
            if a_data["status"] != "success":
                st.error(f"❌ Animation Error: {a_data['error']}")
            if m_data["status"] != "success":
                st.error(f"❌ Metrics Error: {m_data['error']}")














