import asyncio
import logging

import bpy


class AsyncLoopManager:
    def __init__(self, name="async-loop-manager"):
        self.name = name
        self.logger = logging.getLogger(f"AsyncLoopManager.{name}")
        self.loop: asyncio.AbstractEventLoop | None = None
        self.tasks: set[asyncio.Task] = set()
        self._running = False

    @staticmethod
    def _format_key(name):
        key = f"AsyncLoopManager_{name}"
        return key

    @staticmethod
    def register(name="async-loop-manager"):
        key = AsyncLoopManager._format_key(name)

        if key in bpy.app.driver_namespace:
            return bpy.app.driver_namespace[key]

        mgr = AsyncLoopManager(name)
        bpy.app.driver_namespace[key] = mgr

        return mgr

    def unregister(self):
        self.stop()
        key = AsyncLoopManager._format_key(self.name)
        if key in bpy.app.driver_namespace:
            del bpy.app.driver_namespace[key]

    def _maybe_create_loop(self):
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            self.logger.debug(f"Created new event loop")

    def _cancel_all_tasks(self) -> None:
        """Cancels all running tasks."""
        if not self.tasks:
            return

        for task in self.tasks:
            task.cancel()

        if self.loop and not self.loop.is_closed():
            # Run loop one last time to process cancellations
            self.loop.run_until_complete(
                asyncio.gather(*self.tasks, return_exceptions=True)
            )
        self.tasks.clear()

    def _close_loop(self):
        """Closes the event loop and cleans up tasks."""
        if self.loop and not self.loop.is_closed():
            self._cancel_all_tasks()
            self.loop.stop()
            self.loop.close()
            self.loop = None
            self.logger.debug(f"Closed loop")

    def _handle_task_done(self, task: asyncio.Task):
        self.logger.info(f"Task done {task}")
        self.tasks.discard(task)
        try:
            e = task.exception()
        except asyncio.exceptions.CancelledError:
            return
        raise e

    def create_task(self, coro) -> asyncio.Task:
        """Creates and tracks a new task in the loop."""
        if not self.loop:
            self._maybe_create_loop()

        self.logger.info(f"Adding task {coro}")
        task = self.loop.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._handle_task_done)
        return task

    def kick(self):
        """Performs one iteration of the event loop.
        Returns True if there are no more tasks to run."""
        if not self.loop or self.loop.is_closed():
            return True

        if not self._running:
            return True

        # Run one iteration of the loop
        self.loop.call_soon(self.loop.stop)
        self.loop.run_forever()

        # Check if we have any tasks left
        return len(self.tasks) == 0

    @property
    def running(self):
        return self._running

    def start(self):
        """Starts the loop manager."""
        self._running = True
        self._maybe_create_loop()

    def stop(self) -> None:
        """Stops the loop manager and cleans up."""
        self._running = False
        self._close_loop()


class AsyncModalOperatorMixin:
    """
    Usage:
        Inherit from this class and implement async_loop_manager_name and async_execute
    """

    async_loop_manager_name: str = NotImplemented
    timer = None

    async def async_execute(self, context):
        raise NotImplementedError()

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):
        async_loop_manager = AsyncLoopManager.register(self.async_loop_manager_name)

        async_loop_manager.create_task(self.async_execute(context))

        if async_loop_manager.running:
            logging.info(f"AsyncLoopManager {async_loop_manager.name} already running")
            return {"PASS_THROUGH"}

        context.window_manager.modal_handler_add(self)
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.00001, window=context.window)

        async_loop_manager.start()

        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        async_loop_manager = AsyncLoopManager.register(self.async_loop_manager_name)

        stop_after_this_kick = async_loop_manager.kick()

        if stop_after_this_kick:
            async_loop_manager.stop()
            context.window_manager.event_timer_remove(self.timer)
            logging.debug("Stopped asyncio loop kicking")
            return {"FINISHED"}

        return {"RUNNING_MODAL"}
